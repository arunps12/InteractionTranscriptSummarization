import os
from pathlib import Path
from datetime import datetime
import yaml
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
import evaluate
from tqdm import tqdm
from src.interaction_transcript_summarization.logging import logger
from src.interaction_transcript_summarization.entity import ModelEvaluationConfig


class ModelEvaluation:
    """
    Evaluates trained PEGASUS model using ROUGE metrics.
    Computes compression ratio and generates evaluation report.
    """
    
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load ROUGE metric
        self.rouge_metric = evaluate.load("rouge")
    
    def evaluate(self):
        """
        Evaluates the model on test dataset.
        """
        # Load model and tokenizer
        logger.info(f"Loading model from {self.config.model_path}")
        model = AutoModelForSeq2SeqLM.from_pretrained(str(self.config.model_path))
        tokenizer = AutoTokenizer.from_pretrained(str(self.config.tokenizer_name))
        
        model.to(self.device)
        model.eval()
        
        # Load test dataset
        logger.info(f"Loading dataset from {self.config.data_path}")
        dataset = load_from_disk(str(self.config.data_path))
        test_dataset = dataset["test"]
        
        # Generate predictions
        logger.info("Generating predictions on test set...")
        predictions = []
        references = []
        compression_ratios = []
        
        for sample in tqdm(test_dataset, desc="Evaluating"):
            dialogue = sample["dialogue"]
            summary = sample["summary"]
            
            # Tokenize input
            inputs = tokenizer(dialogue, return_tensors="pt", max_length=1024, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate summary
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=4,
                    early_stopping=True
                )
            
            # Decode prediction
            predicted_summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            predictions.append(predicted_summary)
            references.append(summary)
            
            # Calculate compression ratio
            compression_ratio = len(predicted_summary) / len(dialogue) if len(dialogue) > 0 else 0
            compression_ratios.append(compression_ratio)
        
        # Compute ROUGE scores
        logger.info("Computing ROUGE scores...")
        rouge_scores = self.rouge_metric.compute(
            predictions=predictions,
            references=references,
            use_stemmer=True
        )
        
        # Calculate average compression ratio
        avg_compression_ratio = sum(compression_ratios) / len(compression_ratios)
        
        # Prepare results
        results = {
            'rouge1': rouge_scores['rouge1'],
            'rouge2': rouge_scores['rouge2'],
            'rougeL': rouge_scores['rougeL'],
            'compression_ratio': avg_compression_ratio,
            'num_test_samples': len(test_dataset),
        }
        
        logger.info(f"ROUGE-1: {results['rouge1']:.4f}")
        logger.info(f"ROUGE-2: {results['rouge2']:.4f}")
        logger.info(f"ROUGE-L: {results['rougeL']:.4f}")
        logger.info(f"Avg Compression Ratio: {results['compression_ratio']:.4f}")
        
        # Generate evaluation report
        self._generate_evaluation_report(results, predictions[:5], references[:5])
        
        return results
    
    def _generate_evaluation_report(self, results, sample_predictions, sample_references):
        """
        Generates a YAML evaluation report with metrics.
        """
        report_data = {
            'timestamp': self.timestamp,
            'model_path': str(self.config.model_path),
            'tokenizer': self.config.tokenizer_name,
            'device': self.device,
            'metrics': {
                'rouge1': float(results['rouge1']),
                'rouge2': float(results['rouge2']),
                'rougeL': float(results['rougeL']),
                'compression_ratio': float(results['compression_ratio']),
            },
            'test_dataset': {
                'num_samples': results['num_test_samples'],
                'data_path': str(self.config.data_path),
            },
            'sample_predictions': [
                {
                    'prediction': pred,
                    'reference': ref
                }
                for pred, ref in zip(sample_predictions, sample_references)
            ]
        }
        
        report_path = Path(self.config.root_dir) / "evaluation_report.yaml"
        with open(report_path, 'w') as f:
            yaml.dump(report_data, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Evaluation report saved to {report_path}")
    
    def initiate_model_evaluation(self):
        """
        Main entry point for model evaluation pipeline.
        """
        logger.info("="*60)
        logger.info("STAGE 5: Model Evaluation Started")
        logger.info("="*60)
        
        self.evaluate()
        
        logger.info("="*60)
        logger.info("STAGE 5: Model Evaluation Completed")
        logger.info("="*60)
