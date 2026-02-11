from src.interaction_transcript_summarization.config.configuration import ConfigurationManager
from src.interaction_transcript_summarization.components.data_transformation import DataTransformation
from src.interaction_transcript_summarization.logging import logger


class DataTransformationTrainingPipeline:
    """
    Pipeline for Stage 3: Data Transformation
    Orchestrates the data transformation process.
    """
    
    def __init__(self):
        pass

    def initiate_data_transformation(self):
        """
        Executes the data transformation pipeline.
        """
        try:
            config_manager = ConfigurationManager()
            data_transformation_config = config_manager.get_data_transformation_config()
            data_transformation = DataTransformation(config=data_transformation_config)
            data_transformation.initiate_data_transformation()
            
        except Exception as e:
            logger.error(f"Data transformation pipeline failed: {str(e)}")
            raise e