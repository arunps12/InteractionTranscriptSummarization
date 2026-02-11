from src.interaction_transcript_summarization.config.configuration import ConfigurationManager
from src.interaction_transcript_summarization.components.model_evaluation import ModelEvaluation
from src.interaction_transcript_summarization.logging import logger


class ModelEvaluationTrainingPipeline:
    """
    Pipeline for Stage 5: Model Evaluation
    Orchestrates the model evaluation process.
    """
    
    def __init__(self):
        pass

    def initiate_model_evaluation(self):
        """
        Executes the model evaluation pipeline.
        """
        try:
            config_manager = ConfigurationManager()
            model_evaluation_config = config_manager.get_model_evaluation_config()
            model_evaluation = ModelEvaluation(config=model_evaluation_config)
            model_evaluation.initiate_model_evaluation()
            
        except Exception as e:
            logger.error(f"Model evaluation pipeline failed: {str(e)}")
            raise e
