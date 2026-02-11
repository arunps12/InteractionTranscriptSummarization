import os
from pathlib import Path
from datetime import datetime
import yaml
import random
import numpy as np
from src.interaction_transcript_summarization.logging import logger
from transformers import AutoTokenizer
from datasets import load_from_disk
from src.interaction_transcript_summarization.entity import DataTransformationConfig


class DataTransformation:
    """
    Transforms the SAMSum dataset for PEGASUS model training.
    Applies deterministic formatting and tokenization.
    """
    
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set deterministic seeds for reproducibility
        self._set_seeds()
    
    def _set_seeds(self, seed=42):
        """
        Sets random seeds for deterministic behavior.
        """
        random.seed(seed)
        np.random.seed(seed)
        logger.info(f"Set deterministic seeds to {seed}")
    
    def convert_examples_to_features(self, example_batch):
        """
        Tokenizes dialogues and summaries using PEGASUS tokenizer.
        """
        # Tokenize input dialogues
        input_encodings = self.tokenizer(
            example_batch['dialogue'],
            max_length=1024,
            truncation=True,
            padding=False  # Will be handled by data collator during training
        )

        # Tokenize target summaries
        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(
                example_batch['summary'],
                max_length=128,
                truncation=True,
                padding=False
            )

        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }
    
    def convert(self):
        """
        Loads, transforms, and saves the dataset.
        """
        logger.info(f"Loading dataset from {self.config.data_path}...")
        dataset_samsum = load_from_disk(str(self.config.data_path))
        
        logger.info("Applying tokenization...")
        dataset_samsum_pt = dataset_samsum.map(
            self.convert_examples_to_features,
            batched=True,
            desc="Tokenizing dataset"
        )
        
        # Save transformed dataset
        output_path = os.path.join(self.config.root_dir, "samsum_dataset")
        os.makedirs(output_path, exist_ok=True)
        dataset_samsum_pt.save_to_disk(output_path)
        logger.info(f"Transformed dataset saved to {output_path}")
        
        # Generate training manifest
        self._generate_training_manifest(dataset_samsum, dataset_samsum_pt, output_path)
    
    def _generate_training_manifest(self, original_dataset, transformed_dataset, output_path):
        """
        Generates a YAML manifest with transformation metadata.
        """
        manifest_data = {
            'timestamp': self.timestamp,
            'tokenizer': self.config.tokenizer_name,
            'input_data_path': str(self.config.data_path),
            'output_data_path': output_path,
            'transformations': {
                'tokenization': True,
                'max_input_length': 1024,
                'max_target_length': 128,
                'truncation': True,
                'deterministic_seed': 42
            },
            'splits': {
                'train': {
                    'num_samples': len(transformed_dataset['train']),
                    'features': list(transformed_dataset['train'].features.keys())
                },
                'validation': {
                    'num_samples': len(transformed_dataset['validation']),
                    'features': list(transformed_dataset['validation'].features.keys())
                },
                'test': {
                    'num_samples': len(transformed_dataset['test']),
                    'features': list(transformed_dataset['test'].features.keys())
                }
            },
            'sample_input_tokens': len(transformed_dataset['train'][0]['input_ids']),
            'sample_label_tokens': len(transformed_dataset['train'][0]['labels'])
        }
        
        manifest_path = Path(self.config.root_dir) / "training_manifest.yaml"
        with open(manifest_path, 'w') as f:
            yaml.dump(manifest_data, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Training manifest saved to {manifest_path}")
    
    def initiate_data_transformation(self):
        """
        Main entry point for data transformation pipeline.
        """
        logger.info("="*60)
        logger.info("STAGE 3: Data Transformation Started")
        logger.info("="*60)
        
        self.convert()
        
        logger.info("="*60)
        logger.info("STAGE 3: Data Transformation Completed")
        logger.info("="*60)