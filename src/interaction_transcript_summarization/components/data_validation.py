import os
from pathlib import Path
from datetime import datetime
from datasets import load_from_disk
import yaml
from src.interaction_transcript_summarization.logging import logger
from src.interaction_transcript_summarization.entity import DataValidationConfig


class DataValidation:
    """
    Validates the ingested SAMSum dataset.
    Checks schema, empty samples, and length constraints.
    """
    
    def __init__(self, config: DataValidationConfig):
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.validation_results = {
            'timestamp': self.timestamp,
            'data_path': str(config.data_path),
            'checks': {},
            'issues_found': [],
            'status': 'PASSED'
        }
    
    def validate_schema(self, dataset):
        """
        Validates that required columns exist in the dataset.
        """
        required_columns = ['dialogue', 'summary']
        
        for split_name in ['train', 'validation', 'test']:
            split = dataset[split_name]
            columns = split.column_names
            
            missing_columns = [col for col in required_columns if col not in columns]
            
            if missing_columns:
                self.validation_results['issues_found'].append(
                    f"Split '{split_name}': Missing columns {missing_columns}"
                )
                self.validation_results['status'] = 'FAILED'
            
        self.validation_results['checks']['schema_validation'] = {
            'required_columns': required_columns,
            'status': 'PASSED' if not any('Missing columns' in issue for issue in self.validation_results['issues_found']) else 'FAILED'
        }
        
        logger.info("Schema validation completed")
    
    def validate_empty_samples(self, dataset):
        """
        Checks for empty dialogues or summaries.
        """
        empty_dialogues = {'train': 0, 'validation': 0, 'test': 0}
        empty_summaries = {'train': 0, 'validation': 0, 'test': 0}
        
        for split_name in ['train', 'validation', 'test']:
            split = dataset[split_name]
            
            for idx, sample in enumerate(split):
                dialogue = str(sample.get('dialogue', '')).strip()
                summary = str(sample.get('summary', '')).strip()
                
                if not dialogue or len(dialogue) == 0:
                    empty_dialogues[split_name] += 1
                
                if not summary or len(summary) == 0:
                    empty_summaries[split_name] += 1
        
        # Report issues
        total_empty_dialogues = sum(empty_dialogues.values())
        total_empty_summaries = sum(empty_summaries.values())
        
        if total_empty_dialogues > 0:
            self.validation_results['issues_found'].append(
                f"Found {total_empty_dialogues} empty dialogues across splits: {empty_dialogues}"
            )
            self.validation_results['status'] = 'FAILED'
        
        if total_empty_summaries > 0:
            self.validation_results['issues_found'].append(
                f"Found {total_empty_summaries} empty summaries across splits: {empty_summaries}"
            )
            self.validation_results['status'] = 'FAILED'
        
        self.validation_results['checks']['empty_samples'] = {
            'empty_dialogues': empty_dialogues,
            'empty_summaries': empty_summaries,
            'status': 'PASSED' if total_empty_dialogues == 0 and total_empty_summaries == 0 else 'FAILED'
        }
        
        logger.info("Empty samples validation completed")
    
    def validate_length_constraints(self, dataset):
        """
        Validates length constraints on dialogues and summaries.
        """
        min_dialogue_length = 10
        min_summary_length = 5
        max_dialogue_length = 10000
        
        length_violations = {'train': 0, 'validation': 0, 'test': 0}
        compression_violations = {'train': 0, 'validation': 0, 'test': 0}
        
        for split_name in ['train', 'validation', 'test']:
            split = dataset[split_name]
            
            for idx, sample in enumerate(split):
                dialogue = str(sample.get('dialogue', '')).strip()
                summary = str(sample.get('summary', '')).strip()
                
                # Length checks
                if len(dialogue) < min_dialogue_length:
                    length_violations[split_name] += 1
                
                if len(dialogue) > max_dialogue_length:
                    length_violations[split_name] += 1
                
                if len(summary) < min_summary_length:
                    length_violations[split_name] += 1
                
                # Compression check: summary should be shorter than dialogue
                if len(summary) >= len(dialogue):
                    compression_violations[split_name] += 1
        
        total_length_violations = sum(length_violations.values())
        total_compression_violations = sum(compression_violations.values())
        
        if total_length_violations > 0:
            self.validation_results['issues_found'].append(
                f"Found {total_length_violations} length constraint violations: {length_violations}"
            )
            # This is a warning, not a failure
        
        if total_compression_violations > 0:
            self.validation_results['issues_found'].append(
                f"Found {total_compression_violations} compression violations (summary >= dialogue): {compression_violations}"
            )
            # This is a warning, not a failure
        
        self.validation_results['checks']['length_constraints'] = {
            'min_dialogue_length': min_dialogue_length,
            'min_summary_length': min_summary_length,
            'max_dialogue_length': max_dialogue_length,
            'length_violations': length_violations,
            'compression_violations': compression_violations,
            'status': 'PASSED' if total_length_violations == 0 else 'WARNING'
        }
        
        logger.info("Length constraints validation completed")
    
    def validate(self):
        """
        Runs all validation checks.
        """
        logger.info(f"Loading dataset from {self.config.data_path}...")
        dataset = load_from_disk(str(self.config.data_path))
        
        logger.info("Running validation checks...")
        self.validate_schema(dataset)
        self.validate_empty_samples(dataset)
        self.validate_length_constraints(dataset)
        
        # Generate validation report
        self._generate_validation_report()
        
        # Log summary
        logger.info(f"Validation Status: {self.validation_results['status']}")
        if self.validation_results['issues_found']:
            logger.warning(f"Issues found: {len(self.validation_results['issues_found'])}")
            for issue in self.validation_results['issues_found']:
                logger.warning(f"  - {issue}")
        else:
            logger.info("No issues found - dataset is clean!")
    
    def _generate_validation_report(self):
        """
        Generates a YAML validation report.
        """
        report_dir = Path(self.config.root_dir) / self.timestamp
        os.makedirs(report_dir, exist_ok=True)
        
        report_path = report_dir / "validation_report.yaml"
        
        with open(report_path, 'w') as f:
            yaml.dump(self.validation_results, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Validation report saved to {report_path}")
    
    def initiate_data_validation(self):
        """
        Main entry point for data validation pipeline.
        """
        logger.info("="*60)
        logger.info("STAGE 2: Data Validation Started")
        logger.info("="*60)
        
        self.validate()
        
        logger.info("="*60)
        logger.info("STAGE 2: Data Validation Completed")
        logger.info("="*60)
