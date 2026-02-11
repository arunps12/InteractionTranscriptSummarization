import os
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from huggingface_hub import HfApi, create_repo
from src.interaction_transcript_summarization.logging import logger
from src.interaction_transcript_summarization.entity import ModelPusherConfig


class ModelPusher:
    """
    Pushes trained model to HuggingFace Hub.
    Only pushes model_best with tokenizer.
    """
    
    def __init__(self, config: ModelPusherConfig):
        self.config = config
        
        # Never log the token
        if not self.config.hf_token:
            raise ValueError("HF_TOKEN is required but not provided")
        if not self.config.hf_repo_id:
            raise ValueError("HF_REPO_ID is required but not provided")
        
        logger.info(f"Target repository: {self.config.hf_repo_id}")
    
    def push_model_to_hub(self):
        """
        Pushes the best model and tokenizer to HuggingFace Hub.
        """
        try:
            # Initialize HF API
            api = HfApi()
            
            # Create repo if it doesn't exist
            logger.info(f"Ensuring repository {self.config.hf_repo_id} exists...")
            try:
                create_repo(
                    repo_id=self.config.hf_repo_id,
                    token=self.config.hf_token,
                    exist_ok=True,
                    private=False
                )
                logger.info(f"Repository {self.config.hf_repo_id} is ready")
            except Exception as e:
                logger.warning(f"Repository creation/check warning: {e}")
            
            # Load model and tokenizer
            logger.info(f"Loading model from {self.config.model_path}")
            model = AutoModelForSeq2SeqLM.from_pretrained(str(self.config.model_path))
            tokenizer = AutoTokenizer.from_pretrained(str(self.config.tokenizer_name))
            
            # Generate model card
            model_card = self._generate_model_card()
            
            # Save model card
            model_card_path = Path(self.config.model_path) / "README.md"
            with open(model_card_path, 'w') as f:
                f.write(model_card)
            logger.info(f"Model card saved to {model_card_path}")
            
            # Push to hub
            logger.info(f"Pushing model to {self.config.hf_repo_id}...")
            model.push_to_hub(
                repo_id=self.config.hf_repo_id,
                token=self.config.hf_token,
                commit_message="Upload fine-tuned PEGASUS model on SAMSum"
            )
            
            logger.info(f"Pushing tokenizer to {self.config.hf_repo_id}...")
            tokenizer.push_to_hub(
                repo_id=self.config.hf_repo_id,
                token=self.config.hf_token,
                commit_message="Upload tokenizer"
            )
            
            logger.info(f"✅ Model successfully pushed to https://huggingface.co/{self.config.hf_repo_id}")
            
        except Exception as e:
            logger.error(f"Failed to push model to HuggingFace Hub: {str(e)}")
            raise e
    
    def _generate_model_card(self):
        """
        Generates a model card (README.md) for HuggingFace Hub.
        """
        model_card = f"""---
language: en
license: mit
tags:
- summarization
- dialogue-summarization
- pegasus
- samsum
datasets:
- samsum
metrics:
- rouge
widget:
- text: "Hannah: Hey, do you have Betty's number?\\nAmanda: Lemme check\\nHannah: <file_gif>\\nAmanda: Sorry, can't find it."
  example_title: "Example 1"
---

# PEGASUS Fine-tuned on SAMSum Dataset

This model is a fine-tuned version of [google/pegasus-cnn_dailymail](https://huggingface.co/google/pegasus-cnn_dailymail) on the [SAMSum dataset](https://huggingface.co/datasets/samsum) for dialogue summarization.

## Model Description

- **Model:** PEGASUS (Pre-training with Extracted Gap-sentences for Abstractive SUmmarization Sequence-to-sequence)
- **Base Model:** google/pegasus-cnn_dailymail
- **Task:** Dialogue Summarization
- **Dataset:** SAMSum (14,732 training samples)
- **Language:** English
- **License:** MIT 2.0

## Intended Use

This model is designed to generate concise summaries of conversational dialogues, particularly:
- Chat conversations
- Messaging transcripts
- Meeting dialogues
- Customer support interactions

## Training Details

- **Training Framework:** HuggingFace Transformers
- **Optimization:** AdamW
- **Mixed Precision:** FP16 (if GPU available)
- **Experiment Tracking:** MLflow
- **Deterministic Training:** Seed 42

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("{self.config.hf_repo_id}")
model = AutoModelForSeq2SeqLM.from_pretrained("{self.config.hf_repo_id}")

dialogue = \"\"\"
Hannah: Hey, do you have Betty's number?
Amanda: Lemme check
Hannah: <file_gif>
Amanda: Sorry, can't find it.
\"\"\"

inputs = tokenizer(dialogue, return_tensors="pt", max_length=1024, truncation=True)
summary_ids = model.generate(inputs["input_ids"], max_length=128, num_beams=4, early_stopping=True)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print(summary)
# Output: "Hannah asked Amanda for Betty's number but Amanda couldn't find it."
```

## Evaluation Metrics

The model was evaluated on the SAMSum test set using ROUGE metrics:

- **ROUGE-1:** Measures unigram overlap
- **ROUGE-2:** Measures bigram overlap  
- **ROUGE-L:** Measures longest common subsequence

*(Specific scores available in evaluation_report.yaml)*

## Limitations

- Optimized for English dialogue summarization
- May not perform well on:
  - Very long conversations (>1024 tokens)
  - Technical/domain-specific dialogues
  - Non-English text
  - Formal documents (optimized for casual chat)

## Citation

```bibtex
@article{{zhang2019pegasus,
  title={{PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization}},
  author={{Zhang, Jingqing and Zhao, Yao and Saleh, Mohammad and Liu, Peter J}},
  journal={{arXiv preprint arXiv:1912.08777}},
  year={{2019}}
}}

@inproceedings{{gliwa2019samsum,
  title={{SAMSum Corpus: A Human-annotated Dialogue Dataset for Abstractive Summarization}},
  author={{Gliwa, Bogdan and Mochol, Iwona and Biesek, Maciej and Wawer, Aleksander}},
  booktitle={{Proceedings of the 2nd Workshop on New Frontiers in Summarization}},
  pages={{70--79}},
  year={{2019}}
}}
```

## Author

**Arun Prakash Singh**

## License

MIT 2.0

## Acknowledgments

- HuggingFace for the transformers library
- Google for the PEGASUS model
- SAMSum dataset creators
"""
        return model_card
    
    def initiate_model_pusher(self):
        """
        Main entry point for model pusher pipeline.
        """
        logger.info("="*60)
        logger.info("STAGE 6: Push to HuggingFace Hub Started")
        logger.info("="*60)
        
        self.push_model_to_hub()
        
        logger.info("="*60)
        logger.info("STAGE 6: Push to HuggingFace Hub Completed")
        logger.info("="*60)
