# interaction_transcript_summarization

![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![PEGASUS](https://img.shields.io/badge/Model-google%2Fpegasus--cnn__dailymail-blue)
![SAMSum](https://img.shields.io/badge/Dataset-SAMSum-green) ![HF
Hub](https://img.shields.io/badge/Registry-HuggingFaceHub-yellow)
![MLflow](https://img.shields.io/badge/Experiment-MLflow-orange)
![AWS](https://img.shields.io/badge/Cloud-AWS-orange)
![License](https://img.shields.io/badge/License-MIT--2.0-lightgrey)

------------------------------------------------------------------------

## Overview

interaction_transcript_summarization is a modular, artifact-driven text
summarization framework designed for dialogue-style transcripts using:

-   Dataset: SAMSum
-   Base Model: google/pegasus-cnn_dailymail
-   Training Artifacts Root: artifacts/model_trainer
-   Processed Dataset Path: artifacts/data_transformation/samsum_dataset
-   Model Registry: Hugging Face Hub (push fine-tuned weights)
-   Secret handling: Hugging Face token loaded from .env

------------------------------------------------------------------------

## Pipeline Stages

### Stage 1: Data Ingestion

-   Load SAMSum via HuggingFace Datasets
-   Save raw splits
-   Generate data_manifest.yaml
-   Idempotent execution

Artifacts:
artifacts/`<timestamp>`{=html}/data_ingestion/data_manifest.yaml

### Stage 2: Data Validation

-   Validate schema
-   Remove empty samples
-   Length sanity checks
-   Generate validation_report.yaml

Artifacts:
artifacts/`<timestamp>`{=html}/data_validation/validation_report.yaml

### Stage 3: Data Transformation

-   Format dialogue input
-   Tokenize using PEGASUS tokenizer
-   Save processed datasets
-   Generate training_manifest.yaml

Artifacts: artifacts/data_transformation/samsum_dataset/

### Stage 4: Model Trainer

-   HuggingFace Trainer
-   FP16 (if GPU)
-   MLflow logging
-   Save model_best and model_last
-   training_report.yaml

Artifacts: artifacts/model_trainer/

### Stage 5: Evaluation

-   ROUGE-1, ROUGE-2, ROUGE-L
-   evaluation_report.yaml

### Stage 6: Push to Hugging Face Hub

Use .env file:

HF_TOKEN=your_token
HF_REPO_ID=your_username/pegasus-samsum-dialogue-summarizer

Push only model_best + tokenizer.

------------------------------------------------------------------------

Author: Arun Prakash Singh\
License: MIT 2.0
