"""
Main pipeline orchestrator for Interaction Transcript Summarization.
Runs all stages or specific stages based on command-line arguments.
"""
import sys
from src.interaction_transcript_summarization.logging import logger
from src.interaction_transcript_summarization.pipeline.stage_1_data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.interaction_transcript_summarization.pipeline.stage_2_data_validation_pipeline import DataValidationTrainingPipeline
from src.interaction_transcript_summarization.pipeline.stage_2_data_transformation_pipeline import DataTransformationTrainingPipeline
from src.interaction_transcript_summarization.pipeline.stage_4_model_trainer_pipeline import ModelTrainerTrainingPipeline
from src.interaction_transcript_summarization.pipeline.stage_5_model_evaluation_pipeline import ModelEvaluationTrainingPipeline
from src.interaction_transcript_summarization.pipeline.stage_6_model_pusher_pipeline import ModelPusherTrainingPipeline


def run_stage_1():
    """Stage 1: Data Ingestion"""
    STAGE_NAME = "Data Ingestion"
    try:
        logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
        pipeline = DataIngestionTrainingPipeline()
        pipeline.initiate_data_ingestion()
        logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n")
    except Exception as e:
        logger.exception(e)
        raise e


def run_stage_2():
    """Stage 2: Data Validation"""
    STAGE_NAME = "Data Validation"
    try:
        logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
        pipeline = DataValidationTrainingPipeline()
        pipeline.initiate_data_validation()
        logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n")
    except Exception as e:
        logger.exception(e)
        raise e


def run_stage_3():
    """Stage 3: Data Transformation"""
    STAGE_NAME = "Data Transformation"
    try:
        logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
        pipeline = DataTransformationTrainingPipeline()
        pipeline.initiate_data_transformation()
        logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n")
    except Exception as e:
        logger.exception(e)
        raise e


def run_stage_4():
    """Stage 4: Model Training"""
    STAGE_NAME = "Model Training"
    try:
        logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
        pipeline = ModelTrainerTrainingPipeline()
        pipeline.initiate_model_training()
        logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n")
    except Exception as e:
        logger.exception(e)
        raise e


def run_stage_5():
    """Stage 5: Model Evaluation"""
    STAGE_NAME = "Model Evaluation"
    try:
        logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
        pipeline = ModelEvaluationTrainingPipeline()
        pipeline.initiate_model_evaluation()
        logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n")
    except Exception as e:
        logger.exception(e)
        raise e


def run_stage_6():
    """Stage 6: Push to HuggingFace Hub"""
    STAGE_NAME = "Push to HuggingFace Hub"
    try:
        logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
        pipeline = ModelPusherTrainingPipeline()
        pipeline.initiate_model_pusher()
        logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n")
    except Exception as e:
        logger.exception(e)
        raise e


if __name__ == "__main__":
    # Parse command-line arguments
    if len(sys.argv) > 1:
        stage = sys.argv[1]
        
        stage_map = {
            "1": run_stage_1,
            "2": run_stage_2,
            "3": run_stage_3,
            "4": run_stage_4,
            "5": run_stage_5,
            "6": run_stage_6,
        }
        
        if stage in stage_map:
            logger.info(f"Running Stage {stage} only")
            stage_map[stage]()
        elif stage == "all":
            logger.info("Running all stages (1-3, skipping training)")
            run_stage_1()
            run_stage_2()
            run_stage_3()
            logger.info("Stages 1-3 completed. Run 'python main.py 4' to train model.")
        else:
            logger.error(f"Invalid stage: {stage}")
            print("Usage: python main.py [1|2|3|4|5|6|all]")
            print("  1: Data Ingestion")
            print("  2: Data Validation")
            print("  3: Data Transformation")
            print("  4: Model Training")
            print("  5: Model Evaluation")
            print("  6: Push to HuggingFace Hub")
            print("  all: Run stages 1-3 (data pipeline)")
            sys.exit(1)
    else:
        # Default: run stages 1-3 (data pipeline only)
        logger.info("No stage specified, running data pipeline (Stages 1-3)")
        run_stage_1()
        run_stage_2()
        run_stage_3()
        logger.info("\nData pipeline completed!")
        logger.info("Next steps:")
        logger.info("  - To train model: python main.py 4")
        logger.info("  - To evaluate: python main.py 5")
        logger.info("  - To push to HF Hub: python main.py 6")