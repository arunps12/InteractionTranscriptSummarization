# Interaction Transcript Summarization

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/transformers)
[![PEGASUS](https://img.shields.io/badge/Model-PEGASUS-blue?logo=google&logoColor=white)](https://huggingface.co/google/pegasus-cnn_dailymail)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.78-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![DVC](https://img.shields.io/badge/DVC-Pipeline-13ADC7?logo=dvc&logoColor=white)](https://dvc.org)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org)
[![DagsHub](https://img.shields.io/badge/DagsHub-Experiments-FF6F61)](https://dagshub.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://hub.docker.com)
[![CI/CD](https://img.shields.io/badge/GitHub_Actions-CI%2FCD-2088FF?logo=githubactions&logoColor=white)](https://github.com/features/actions)
[![UV](https://img.shields.io/badge/UV-Package_Manager-DE5FE9?logo=astral&logoColor=white)](https://docs.astral.sh/uv)
[![Dataset: SAMSum](https://img.shields.io/badge/Dataset-SAMSum-green)](https://huggingface.co/datasets/Samsung/samsum)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-ready, modular dialogue summarization system using fine-tuned PEGASUS on the SAMSum dataset, with complete MLOps pipeline including training, evaluation, deployment, and CI/CD.

## Features

- **Complete ML Pipeline**: 6-stage modular pipeline from data ingestion to model deployment
- **DVC Integration**: Smart pipeline caching - only re-runs stages when dependencies change
- **PEGASUS Fine-tuning**: State-of-the-art abstractive summarization
- **MLflow Integration**: Comprehensive experiment tracking
- **FastAPI Service**: REST API for inference with latency tracking
- **Docker Support**: Containerized deployment
- **CI/CD**: GitHub Actions workflow with AWS deployment
- **HuggingFace Hub**: Model registry and versioning

## Pipeline Stages

### Stage 1: Data Ingestion
- Loads SAMSum dataset from HuggingFace
- Idempotent execution
- Generates data manifest with statistics

### Stage 2: Data Validation
- Schema validation
- Empty sample detection
- Length constraint checks
- Compression ratio validation

### Stage 3: Data Transformation
- Deterministic PEGASUS tokenization (seed=42)
- Max input: 1024 tokens
- Max target: 128 tokens
- Training manifest generation

### Stage 4: Model Training
- HuggingFace Trainer with FP16 support
- MLflow experiment tracking
- Saves best and last checkpoints
- Training report generation

### Stage 5: Model Evaluation
- ROUGE-1, ROUGE-2, ROUGE-L metrics
- Compression ratio analysis
- Evaluation report with sample predictions

### Stage 6: Push to HuggingFace Hub
- Uploads best model + tokenizer
- Auto-generates model card
- Environment-based authentication

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd InteractionTranscriptSummarization

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt

# Initialize DVC
dvc init

# Set up environment variables
cp .env.example .env
# Edit .env with your HuggingFace token
```

### Running the Pipeline

#### Using DVC (Recommended - Smart Caching)

```bash
# Run entire pipeline with DVC (only re-runs changed stages)
uv run dvc repro

# Run specific stage and its dependencies
uv run dvc repro data_ingestion
uv run dvc repro data_validation
uv run dvc repro data_transformation
uv run dvc repro model_trainer
uv run dvc repro model_evaluation
uv run dvc repro model_pusher

# Force re-run a specific stage
uv run dvc repro -f model_trainer
```

#### Using Python Directly

```bash
# Run entire data pipeline (Stages 1-3)
uv run python main.py

# Run specific stages
uv run python main.py 1  # Data Ingestion
uv run python main.py 2  # Data Validation
uv run python main.py 3  # Data Transformation
uv run python main.py 4  # Model Training (requires GPU recommended)
uv run python main.py 5  # Model Evaluation
uv run python main.py 6  # Push to HuggingFace Hub
```

### Running the API

```bash
# Start the FastAPI server
uv run uvicorn app:app --reload

# Or using Python directly
uv run python app.py
```

Visit `http://localhost:8000/docs` for interactive API documentation.

### API Usage

```python
import requests

response = requests.post(
    "http://localhost:8000/summarize",
    json={
        "text": "Hannah: Hey, do you have Betty's number?\nAmanda: Lemme check\nAmanda: Sorry, can't find it."
    }
)

print(response.json())
# Output:
# {
#   "summary": "Hannah asked Amanda for Betty's number but Amanda couldn't find it.",
#   "model": "pegasus-samsum-local",
#   "version": "1.0.0",
#   "latency_ms": 245.3
# }
```

## Docker Deployment

```bash
# Build the Docker image
docker build -t pegasus-summarizer .

# Run the container
docker run -p 8000:8000 \
  -e HF_TOKEN=your_token \
  -e HF_REPO_ID=your_username/repo \
  pegasus-summarizer
```

## Project Structure

```
├── src/interaction_transcript_summarization/
│   ├── components/          # Pipeline components
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluation.py
│   │   └── model_pusher.py
│   ├── pipeline/            # Pipeline orchestrators
│   ├── config/              # Configuration management
│   ├── entity/              # Data classes
│   ├── utils/               # Utilities
│   └── logging/             # Logging setup
├── config/
│   └── config.yaml          # Pipeline configuration
├── dvc.yaml                 # DVC pipeline definition
├── params.yaml              # Training hyperparameters
├── main.py                  # Pipeline runner
├── app.py                   # FastAPI application
├── Dockerfile               # Container definition
├── requirements.txt         # Dependencies
└── .github/workflows/       # CI/CD workflows
```

## Configuration

### Environment Variables (.env)

```bash
HF_TOKEN=your_huggingface_token
HF_REPO_ID=username/pegasus-samsum-dialogue-summarizer
```

### Training Parameters (params.yaml)

```yaml
TrainingArguments:
  num_train_epochs: 1
  learning_rate: 5.0e-5
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  fp16: true
```

## Model Performance

The fine-tuned model achieves:
- **ROUGE-1**: ~0.47
- **ROUGE-2**: ~0.24
- **ROUGE-L**: ~0.39

*(Actual scores depend on training configuration)*

## Testing

```bash
# Run tests
uv run pytest tests/ -v

# Run linting
uv run ruff check .

# Check DVC pipeline status
uv run dvc status

# Visualize DVC pipeline DAG
uv run dvc dag
```

## CI/CD Pipeline

The GitHub Actions workflow automatically:
1. Runs linting (Ruff) and tests (pytest)
2. Builds Docker image
3. Pushes to Amazon ECR
4. Deploys to AWS (ECS/EC2)

## License

MIT License - see LICENSE file for details

## Author

**Arun Prakash Singh**

## Acknowledgments

- HuggingFace for the Transformers library
- Google for the PEGASUS model
- SAMSum dataset creators
- MLflow for experiment tracking

## Documentation

- [Detailed Specification](docs/interaction_transcript_summarization_README_AWS_GHA_HFHub.md)
- [Refactoring Report](docs/refactor_report.md)
- [Assumptions](docs/assumptions.md)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or support, please open an issue on GitHub.
