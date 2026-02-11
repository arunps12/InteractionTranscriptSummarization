from src.interaction_transcript_summarization.config.configuration import ConfigurationManager
from src.interaction_transcript_summarization.components.data_validation import DataValidation
from src.interaction_transcript_summarization.logging import logger


class DataValidationTrainingPipeline:
    """
    Pipeline for Stage 2: Data Validation
    Orchestrates the data validation process.
    """
    
    def __init__(self):
        pass

    def initiate_data_validation(self):
        """
        Executes the data validation pipeline.
        """
        try:
            config_manager = ConfigurationManager()
            data_validation_config = config_manager.get_data_validation_config()
            data_validation = DataValidation(config=data_validation_config)
            data_validation.initiate_data_validation()
            
        except Exception as e:
            logger.error(f"Data validation pipeline failed: {str(e)}")
            raise e
