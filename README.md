# Interaction Transcript Summarization

![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![PEGASUS](https://img.shields.io/badge/Model-PEGASUS-blue)
![SAMSum](https://img.shields.io/badge/Dataset-SAMSum-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A production-ready, modular dialogue summarization system using fine-tuned PEGASUS on the SAMSum dataset, with complete MLOps pipeline including training, evaluation, deployment, and CI/CD.

## 🎯 Features

- **Complete ML Pipeline**: 6-stage modular pipeline from data ingestion to model deployment
- **PEGASUS Fine-tuning**: State-of-the-art abstractive summarization
- **MLflow Integration**: Comprehensive experiment tracking
- **FastAPI Service**: REST API for inference with latency tracking
- **Docker Support**: Containerized deployment
- **CI/CD**: GitHub Actions workflow with AWS deployment
- **HuggingFace Hub**: Model registry and versioning

## 📋 Pipeline Stages

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

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd InteractionTranscriptSummarization

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your HuggingFace token
```

### Running the Pipeline

```bash
# Run entire data pipeline (Stages 1-3)
python main.py

# Run specific stages
python main.py 1  # Data Ingestion
python main.py 2  # Data Validation
python main.py 3  # Data Transformation
python main.py 4  # Model Training (requires GPU recommended)
python main.py 5  # Model Evaluation
python main.py 6  # Push to HuggingFace Hub
```

### Running the API

```bash
# Start the FastAPI server
uvicorn app:app --reload

# Or using Python
python app.py
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

## 🐳 Docker Deployment

```bash
# Build the Docker image
docker build -t pegasus-summarizer .

# Run the container
docker run -p 8000:8000 \
  -e HF_TOKEN=your_token \
  -e HF_REPO_ID=your_username/repo \
  pegasus-summarizer
```

## 📊 Project Structure

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
├── params.yaml              # Training hyperparameters
├── main.py                  # Pipeline runner
├── app.py                   # FastAPI application
├── Dockerfile               # Container definition
├── requirements.txt         # Dependencies
└── .github/workflows/       # CI/CD workflows
```

## 🔧 Configuration

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

## 📈 Model Performance

The fine-tuned model achieves:
- **ROUGE-1**: ~0.47
- **ROUGE-2**: ~0.24
- **ROUGE-L**: ~0.39

*(Actual scores depend on training configuration)*

## 🧪 Testing

```bash
# Run tests
pytest tests/ -v

# Run linting
ruff check .
```

## 🚢 CI/CD Pipeline

The GitHub Actions workflow automatically:
1. Runs linting (Ruff) and tests (pytest)
2. Builds Docker image
3. Pushes to Amazon ECR
4. Deploys to AWS (ECS/EC2)

## 📝 License

MIT License - see LICENSE file for details

## 👤 Author

**Arun Prakash Singh**

## 🙏 Acknowledgments

- HuggingFace for the Transformers library
- Google for the PEGASUS model
- SAMSum dataset creators
- MLflow for experiment tracking

## 📚 Documentation

- [Detailed Specification](docs/interaction_transcript_summarization_README_AWS_GHA_HFHub.md)
- [Refactoring Report](docs/refactor_report.md)
- [Assumptions](docs/assumptions.md)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or support, please open an issue on GitHub.
