from src.interaction_transcript_summarization.config.configuration import ConfigurationManager
from src.interaction_transcript_summarization.components.data_transformation import DataTransformation
from src.interaction_transcript_summarization.logging import logger


class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def initiate_data_transformation(self):
        config=ConfigurationManager()
        data_transformation_config=config.get_data_transformation_config()
        data_transformation=DataTransformation(config=data_transformation_config)
        data_transformation.convert()
        