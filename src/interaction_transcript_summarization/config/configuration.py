from src.interaction_transcript_summarization.constants import *
from src.interaction_transcript_summarization.utils.common import read_yaml, create_directories
from src.interaction_transcript_summarization.entity import (
    DataIngestionConfig, 
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelPusherConfig
)
from pathlib import Path
import os
from dotenv import load_dotenv

class ConfigurationManager:
    def __init__(
        self,
        config_path: Path = CONFIG_FILE_PATH,
        params_path: Path = PARAMS_FILE_PATH
    ):
        self.config = read_yaml(config_path)
        self.params = read_yaml(params_path)
        
        create_directories([self.config.artifacts_root])
        
        # Load environment variables
        load_dotenv()

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            dataset_name=config.dataset_name,
            raw_data_path=Path(config.raw_data_path)
        )
        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        create_directories([config.root_dir])
        
        data_validation_config = DataValidationConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path)
        )
        return data_validation_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            tokenizer_name=config.tokenizer_name
        )
        return data_transformation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.TrainingArguments
        
        create_directories([config.root_dir, config.model_best_path, config.model_last_path])
        
        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            model_ckpt=config.model_ckpt,
            model_best_path=Path(config.model_best_path),
            model_last_path=Path(config.model_last_path),
            num_train_epochs=params.num_train_epochs,
            learning_rate=params.learning_rate,
            warmup_steps=params.warmup_steps,
            weight_decay=params.weight_decay,
            per_device_train_batch_size=params.per_device_train_batch_size,
            per_device_eval_batch_size=params.per_device_eval_batch_size,
            gradient_accumulation_steps=params.gradient_accumulation_steps,
            fp16=params.fp16,
            eval_strategy=params.eval_strategy,
            eval_steps=params.eval_steps,
            logging_steps=params.logging_steps,
            predict_with_generate=params.predict_with_generate,
            generation_max_length=params.generation_max_length,
            save_strategy=params.save_strategy,
            save_steps=params.save_steps
        )
        return model_trainer_config
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        create_directories([config.root_dir])
        
        model_evaluation_config = ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            model_path=Path(config.model_path),
            data_path=Path(config.data_path),
            tokenizer_name=config.tokenizer_name
        )
        return model_evaluation_config
    
    def get_model_pusher_config(self) -> ModelPusherConfig:
        config = self.config.model_pusher
        
        # Get secrets from environment variables
        hf_token = os.getenv("HF_TOKEN")
        hf_repo_id = os.getenv("HF_REPO_ID")
        
        if not hf_token:
            raise ValueError("HF_TOKEN not found in environment variables. Please set it in .env file")
        if not hf_repo_id:
            raise ValueError("HF_REPO_ID not found in environment variables. Please set it in .env file")
        
        model_pusher_config = ModelPusherConfig(
            model_path=Path(config.model_path),
            tokenizer_name=config.tokenizer_name,
            hf_token=hf_token,
            hf_repo_id=hf_repo_id
        )
        return model_pusher_config