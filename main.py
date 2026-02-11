from src.interaction_transcript_summarization.logging import logger
from src.interaction_transcript_summarization.pipeline.stage_1_data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.interaction_transcript_summarization.pipeline.stage_2_data_validation_pipeline import DataValidationTrainingPipeline

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


# Stage 2: Data Validation
STAGE_NAME = "Data Validation"

try:
    logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
    pipeline = DataValidationTrainingPipeline()
    pipeline.initiate_data_validation()
    logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n")
except Exception as e:
    logger.exception(e)
    raise e