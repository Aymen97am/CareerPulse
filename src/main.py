from typing import List
from datetime import datetime
import os
import json
import logging
import pandas as pd

from career_pulse.process_job import JobProcessingChain
from career_pulse.structures import EnrichedJob
from career_pulse.constants import PROJECT_ROOT, REPORTS_DIR, DATA_DIR


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def save_enriched_jobs(enriched_jobs: List[EnrichedJob]):
    """Save enriched jobs to JSON and CSV formats."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)

    # Save as JSON (full data)
    json_path = os.path.join(REPORTS_DIR, f"enriched_jobs_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump([job.model_dump() for job in enriched_jobs], f, indent=2)
    logger.info(f"Saved JSON to {json_path}")

    return json_path


def main():
    # Example: Process jobs from CSV
    csv_path = os.path.join(DATA_DIR, "example.csv")

    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        exit(1)

    # Load jobs
    jobs_df = pd.read_csv(csv_path)  # .head(1)
    jobs_df = jobs_df.dropna(subset=["company", "description"])
    logger.info(f"Loaded {len(jobs_df)} jobs from {csv_path}")

    # Process jobs (with parallelism)
    processor = JobProcessingChain()  # model="Qwen/Qwen3-4B-Instruct-2507")
    enriched_jobs = processor.process_jobs_parallel(jobs_df, batch_size=20)

    # Filter to technical jobs only
    technical_jobs = [job for job in enriched_jobs if job.classification.is_technical]

    logger.info(
        f"\n{'='*60}\nResults: {len(technical_jobs)}/{len(enriched_jobs)} technical jobs found\n{'='*60}\n"
    )

    # Save results
    json_path = save_enriched_jobs(enriched_jobs)

    logger.info(f"\n✅ Processing complete!")
    logger.info(f"📄 Results saved to {json_path}")


if __name__ == "__main__":
    main()
