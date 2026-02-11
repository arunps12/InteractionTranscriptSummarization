from src.interaction_transcript_summarization.config.configuration import ConfigurationManager
from src.interaction_transcript_summarization.components.model_trainer import ModelTrainer
from src.interaction_transcript_summarization.logging import logger


class ModelTrainerTrainingPipeline:
    """
    Pipeline for Stage 4: Model Training
    Orchestrates the model training process.
    """
    
    def __init__(self):
        pass

    def initiate_model_training(self):
        """
        Executes the model training pipeline.
        """
        try:
            config_manager = ConfigurationManager()
            model_trainer_config = config_manager.get_model_trainer_config()
            model_trainer = ModelTrainer(config=model_trainer_config)
            model_trainer.initiate_model_training()
            
        except Exception as e:
            logger.error(f"Model training pipeline failed: {str(e)}")
            raise e
