import os
import torch
from pathlib import Path
from datetime import datetime
import yaml
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import load_from_disk
import mlflow
from src.interaction_transcript_summarization.logging import logger
from src.interaction_transcript_summarization.entity import ModelTrainerConfig


class ModelTrainer:
    """
    Trains PEGASUS model on SAMSum dataset.
    Integrates with MLflow for experiment tracking.
    """
    
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set deterministic seeds
        self._set_seeds()
        
        # Check GPU availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fp16_enabled = self.config.fp16 and torch.cuda.is_available()
        
        logger.info(f"Device: {self.device}")
        logger.info(f"FP16 enabled: {self.fp16_enabled}")
    
    def _set_seeds(self, seed=42):
        """Sets random seeds for reproducibility."""
        import random
        import numpy as np
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        logger.info(f"Set deterministic seeds to {seed}")
    
    def train(self):
        """
        Main training method.
        """
        # Load tokenizer and model
        logger.info(f"Loading tokenizer and model: {self.config.model_ckpt}")
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt)
        
        # Load dataset
        logger.info(f"Loading tokenized dataset from {self.config.data_path}")
        dataset = load_from_disk(str(self.config.data_path))
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        
        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(self.config.root_dir),
            num_train_epochs=self.config.num_train_epochs,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            fp16=self.fp16_enabled,
            evaluation_strategy=self.config.eval_strategy,
            eval_steps=self.config.eval_steps,
            logging_steps=self.config.logging_steps,
            save_strategy=self.config.save_strategy,
            save_steps=self.config.save_steps,
            predict_with_generate=self.config.predict_with_generate,
            generation_max_length=self.config.generation_max_length,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=2,
            report_to=["mlflow"],
            logging_dir=os.path.join(self.config.root_dir, "logs"),
        )
        
        # Initialize MLflow
        mlflow.set_experiment("pegasus-samsum-training")
        
        # Trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"training_{self.timestamp}"):
            # Log parameters
            mlflow.log_params({
                "model_checkpoint": self.config.model_ckpt,
                "num_train_epochs": self.config.num_train_epochs,
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.per_device_train_batch_size,
                "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                "fp16": self.fp16_enabled,
                "device": self.device,
            })
            
            # Train
            logger.info("Starting training...")
            train_result = trainer.train()
            
            # Save best model
            logger.info(f"Saving best model to {self.config.model_best_path}")
            trainer.save_model(str(self.config.model_best_path))
            tokenizer.save_pretrained(str(self.config.model_best_path))
            
            # Save last model
            logger.info(f"Saving last model to {self.config.model_last_path}")
            model.save_pretrained(str(self.config.model_last_path))
            tokenizer.save_pretrained(str(self.config.model_last_path))
            
            # Log metrics
            mlflow.log_metrics({
                "train_loss": train_result.training_loss,
                "train_samples": len(dataset["train"]),
            })
            
            # Generate training report
            self._generate_training_report(train_result, trainer)
            
            logger.info("Training completed successfully")
    
    def _generate_training_report(self, train_result, trainer):
        """
        Generates a YAML training report with metrics and metadata.
        """
        report_data = {
            'timestamp': self.timestamp,
            'model_checkpoint': self.config.model_ckpt,
            'device': self.device,
            'fp16_enabled': self.fp16_enabled,
            'training_arguments': {
                'num_train_epochs': self.config.num_train_epochs,
                'learning_rate': self.config.learning_rate,
                'warmup_steps': self.config.warmup_steps,
                'weight_decay': self.config.weight_decay,
                'per_device_train_batch_size': self.config.per_device_train_batch_size,
                'per_device_eval_batch_size': self.config.per_device_eval_batch_size,
                'gradient_accumulation_steps': self.config.gradient_accumulation_steps,
            },
            'training_results': {
                'training_loss': float(train_result.training_loss),
                'global_step': train_result.global_step,
                'training_samples': train_result.metrics.get('train_samples', 'N/A'),
            },
            'model_paths': {
                'best_model': str(self.config.model_best_path),
                'last_model': str(self.config.model_last_path),
            },
            'experiment_tracking': 'MLflow',
        }
        
        report_path = Path(self.config.root_dir) / "training_report.yaml"
        with open(report_path, 'w') as f:
            yaml.dump(report_data, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Training report saved to {report_path}")
    
    def initiate_model_training(self):
        """
        Main entry point for model training pipeline.
        """
        logger.info("="*60)
        logger.info("STAGE 4: Model Training Started")
        logger.info("="*60)
        
        self.train()
        
        logger.info("="*60)
        logger.info("STAGE 4: Model Training Completed")
        logger.info("="*60)
