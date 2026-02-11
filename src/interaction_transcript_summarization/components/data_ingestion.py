import os
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
import yaml
from src.interaction_transcript_summarization.logging import logger
from src.interaction_transcript_summarization.entity import DataIngestionConfig


class DataIngestion:
    """
    Handles data ingestion from HuggingFace datasets library.
    Loads SAMSum dataset and saves to local disk with metadata.
    """
    
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def download_and_save_dataset(self):
        """
        Downloads SAMSum dataset from HuggingFace and saves to disk.
        Also generates data manifest with metadata.
        Idempotent: skips if dataset already exists.
        """
        try:
            # Check if dataset already exists
            if os.path.exists(self.config.raw_data_path):
                logger.info(f"Dataset already exists at {self.config.raw_data_path}")
                logger.info("Loading existing dataset for manifest generation...")
                from datasets import load_from_disk
                dataset = load_from_disk(str(self.config.raw_data_path))
            else:
                logger.info(f"Loading {self.config.dataset_name} dataset from HuggingFace...")
                
                # Load SAMSum dataset
                # Try multiple possible dataset names
                try:
                    dataset = load_dataset(self.config.dataset_name)
                except Exception:
                    logger.warning(f"Failed to load '{self.config.dataset_name}', trying 'samsum'...")
                    try:
                        dataset = load_dataset("samsum")
                    except Exception:
                        logger.error("Could not load SAMSum dataset from HuggingFace Hub")
                        raise
                
                logger.info(f"Dataset loaded successfully:")
                logger.info(f"  - Train: {len(dataset['train'])} samples")
                logger.info(f"  - Validation: {len(dataset['validation'])} samples")
                logger.info(f"  - Test: {len(dataset['test'])} samples")
                
                # Save dataset to disk
                os.makedirs(self.config.raw_data_path, exist_ok=True)
                dataset.save_to_disk(str(self.config.raw_data_path))
                logger.info(f"Dataset saved to {self.config.raw_data_path}")
            
            # Generate data manifest
            self._generate_data_manifest(dataset)
            
            logger.info("Data ingestion completed successfully")
            
        except Exception as e:
            logger.error(f"Error during data ingestion: {str(e)}")
            raise e
    
    def _generate_data_manifest(self, dataset):
        """
        Generates a YAML manifest with dataset metadata.
        """
        manifest_dir = Path(self.config.root_dir) / self.timestamp
        os.makedirs(manifest_dir, exist_ok=True)
        
        manifest_path = manifest_dir / "data_manifest.yaml"
        
        # Collect statistics
        train_sample = dataset['train'][0]
        columns = list(train_sample.keys())
        
        manifest_data = {
            'dataset_name': self.config.dataset_name,
            'timestamp': self.timestamp,
            'source': 'HuggingFace Datasets',
            'raw_data_path': str(self.config.raw_data_path),
            'splits': {
                'train': {
                    'num_samples': len(dataset['train']),
                    'columns': columns
                },
                'validation': {
                    'num_samples': len(dataset['validation']),
                    'columns': columns
                },
                'test': {
                    'num_samples': len(dataset['test']),
                    'columns': columns
                }
            },
            'sample_dialogue': train_sample.get('dialogue', 'N/A'),
            'sample_summary': train_sample.get('summary', 'N/A')
        }
        
        with open(manifest_path, 'w') as f:
            yaml.dump(manifest_data, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Data manifest saved to {manifest_path}")
        
    def initiate_data_ingestion(self):
        """
        Main entry point for data ingestion pipeline.
        """
        logger.info("="*60)
        logger.info("STAGE 1: Data Ingestion Started")
        logger.info("="*60)
        
        self.download_and_save_dataset()
        
        logger.info("="*60)
        logger.info("STAGE 1: Data Ingestion Completed")
        logger.info("="*60)
