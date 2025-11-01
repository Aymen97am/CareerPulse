"""
Advanced job classification and extraction using LangChain with:
- Conditional routing (filter non-technical jobs)
- Parallel extraction (multiple fields at once)
- Structured outputs (Pydantic models)
"""

from typing import List
import pandas as pd
import logging
import os


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableParallel, RunnableLambda

from career_pulse.structures import EnrichedJob, TechnicalClass, GenericJobInfo
from career_pulse.constants import PROMPTS_DIR

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Environment variables
from dotenv import load_dotenv

load_dotenv()

# LLM configuration
MODEL_CONFIGS = {
    "gemini-2.0-flash-lite": {
        "provider": "google",
        "temperature": 0.01,
    },
    "Qwen/Qwen3-4B-Instruct-2507": {
        "provider": "huggingface",
        "temperature": 0.01,
    },
    "Qwen/Qwen3-30B-A3B-Instruct-2507": {
        "provider": "huggingface",
        "temperature": 0.01,
    },
}


class JobProcessingChain:
    """
    LangChain pipeline that:
    1. Classifies if job is technical
    2. If technical → extracts job info in parallel
    3. If not technical → skips extraction
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash-lite",
        temperature: float = 0.01,
        classifier_version: str = "v1",
        extractor_version: str = "v1",
    ):
        self.llm = self._setup_llm(model=model, temperature=temperature)

        self.classification_chain = self._create_classification_chain(
            prompt_version=classifier_version
        )
        self.extraction_chain = self._create_extraction_chain(
            prompt_version=extractor_version
        )
        self.full_pipeline = self._create_full_pipeline()

    def process_jobs_parallel(
        self, df: pd.DataFrame, batch_size: int = 10
    ) -> List[EnrichedJob]:
        """
        Process jobs in parallel batches for faster execution.

        Each batch processes multiple jobs simultaneously.
        """
        jobs_dicts = df.to_dict("records")
        results = []

        for i in range(0, len(jobs_dicts), batch_size):
            batch = jobs_dicts[i : i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1} ({len(batch)} jobs)...")

            try:
                # Batch processing - all jobs in parallel!
                batch_results = self.full_pipeline.batch(batch)
                results.extend(batch_results)
            except Exception as e:
                logger.error(f"Error in batch processing: {str(e)}")
                # Fallback to sequential for this batch
                for job in batch:
                    try:
                        result = self.full_pipeline.invoke(job)
                        results.append(result)
                    except Exception as e2:
                        logger.error(f"Error processing job: {str(e2)}")
                        continue

        return results

    def _create_full_pipeline(self):
        """
        Create the full conditional pipeline:

        Job → Classify → Branch:
                         ├─ If Technical: Extract info
                         └─ If Not Technical: Return None for extraction
        """

        format_input = RunnableLambda(
            lambda job: {
                "title": job.get("title"),
                "company": job.get("company"),
                "location": job.get("location"),
                "description": job.get("description"),
                "classification_instructions": self.class_parser.get_format_instructions(),
                "extraction_instructions": self.extraction_parser.get_format_instructions(),
            }
        )

        # Assemble full pipeline
        full_pipeline = (
            format_input
            | self.classification_chain
            | self.extraction_chain
            | RunnableLambda(lambda output: EnrichedJob(**output))
        )
        return full_pipeline

    def _create_classification_chain(self, prompt_version: str):
        """
        Chain to classify if a job is technical.

        Job Description → Classify → TechnicalClass
        """
        self.class_parser = PydanticOutputParser(pydantic_object=TechnicalClass)
        template = open(
            os.path.join(PROMPTS_DIR, f"classification_{prompt_version}.txt")
        ).read()
        prompt = ChatPromptTemplate.from_template(template)

        classification_chain = RunnableParallel(
            # Keep full job context to be passed to extraction chain
            job=lambda x: x,
            classification=prompt | self.llm | self.class_parser,
        ) | RunnableLambda(
            lambda x: {**x["job"], "classification": x["classification"]}
        )  # Flatten output dict
        return classification_chain

    def _create_extraction_chain(self, prompt_version: str):
        """
        Chain to extract structured info from technical jobs.

        Job Description → Extract (Parallel) → GenericJobInfo
        """
        self.extraction_parser = PydanticOutputParser(pydantic_object=GenericJobInfo)
        template = open(
            os.path.join(PROMPTS_DIR, f"extraction_{prompt_version}.txt")
        ).read()

        prompt = ChatPromptTemplate.from_template(template)

        # Conditional extraction
        extraction_chain = RunnableParallel(
            # Keep outputs from previous step
            input=lambda x: x,
            extraction=RunnableBranch(
                # Only extract if job is technical
                (
                    lambda x: x["classification"].is_technical,
                    prompt | self.llm | self.extraction_parser,
                ),
                # Default case (non-technical)
                RunnableLambda(lambda _: None),
            ),
        ) | RunnableLambda(
            lambda x: {**x["input"], "extraction": x["extraction"]}
        )  # Flatten output dict

        return extraction_chain

    def _setup_llm(self, model: str, temperature: float) -> BaseChatModel:
        """
        Set up the appropriate LangChain LLM based on the model configuration.

        Returns:
            BaseChatModel: Configured LangChain chat model
        """
        if model not in MODEL_CONFIGS:
            raise ValueError(
                f"Model {model} not supported. "
                f"Available models: {list(MODEL_CONFIGS.keys())}"
            )

        provider = MODEL_CONFIGS[model]["provider"]

        logger.info(f"Setting up {provider} model: {model}")

        if provider == "google":
            return ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
            )
        elif provider == "huggingface":
            llm = HuggingFaceEndpoint(
                repo_id=model,
                temperature=temperature,
                huggingfacehub_api_token=os.getenv("HF_TOKEN"),
            )
            return ChatHuggingFace(llm=llm)
        else:
            raise ValueError(f"Unknown provider: {provider}")
