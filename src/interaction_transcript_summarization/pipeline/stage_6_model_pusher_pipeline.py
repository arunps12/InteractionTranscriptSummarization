from src.interaction_transcript_summarization.config.configuration import ConfigurationManager
from src.interaction_transcript_summarization.components.model_pusher import ModelPusher
from src.interaction_transcript_summarization.logging import logger


class ModelPusherTrainingPipeline:
    """
    Pipeline for Stage 6: Push to HuggingFace Hub
    Orchestrates the model pushing process.
    """
    
    def __init__(self):
        pass

    def initiate_model_pusher(self):
        """
        Executes the model pusher pipeline.
        """
        try:
            config_manager = ConfigurationManager()
            model_pusher_config = config_manager.get_model_pusher_config()
            model_pusher = ModelPusher(config=model_pusher_config)
            model_pusher.initiate_model_pusher()
            
        except Exception as e:
            logger.error(f"Model pusher pipeline failed: {str(e)}")
            raise e
