from src.interaction_transcript_summarization.config.configuration import ConfigurationManager
from src.interaction_transcript_summarization.components.data_ingestion import DataIngestion
from src.interaction_transcript_summarization.logging import logger


class DataIngestionTrainingPipeline:
    """
    Pipeline for Stage 1: Data Ingestion
    Orchestrates the data ingestion process.
    """
    
    def __init__(self):
        pass

    def initiate_data_ingestion(self):
        """
        Executes the data ingestion pipeline.
        """
        try:
            config_manager = ConfigurationManager()
            data_ingestion_config = config_manager.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_ingestion.initiate_data_ingestion()
            
        except Exception as e:
            logger.error(f"Data ingestion pipeline failed: {str(e)}")
            raise e