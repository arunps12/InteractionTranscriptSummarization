from src.interaction_transcript_summarization.logging import logger
from src.interaction_transcript_summarization.pipeline.stage_1_data_ingestion_pipeline import DataIngestionTrainingPipeline

# Stage 1: Data Ingestion
STAGE_NAME = "Data Ingestion"

try:
    logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
    pipeline = DataIngestionTrainingPipeline()
    pipeline.initiate_data_ingestion()
    logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n")
except Exception as e:
    logger.exception(e)
    raise e