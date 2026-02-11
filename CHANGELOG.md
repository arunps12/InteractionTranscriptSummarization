# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-02-11

### Added

#### Phase 0 - Structure Refactoring
- Project structure refactored to match template.py specification
- Environment-based secret management (.env support)
- Comprehensive documentation (refactor_report.md, assumptions.md)
- Updated .gitignore to protect sensitive files while allowing essential docs
- Configuration entities updated with frozen dataclasses for immutability

#### Stage 1 - Data Ingestion
- SAMSum dataset loading from HuggingFace Datasets
- Idempotent data ingestion (skips if dataset exists)
- Timestamped data_manifest.yaml generation
- Support for 14,732 training + 818 validation + 819 test samples

#### Stage 2 - Data Validation
- Comprehensive schema validation (dialogue + summary columns)
- Empty sample detection
- Length constraint validation (min/max checks)
- Compression ratio validation (summary vs dialogue length)
- Timestamped validation_report.yaml with detailed statistics

#### Stage 3 - Data Transformation
- Deterministic PEGASUS tokenization with seed control (seed=42)
- Max input: 1024 tokens, max target: 128 tokens
- Batched tokenization for efficiency
- training_manifest.yaml with transformation metadata

#### Stage 4 - Model Training
- HuggingFace Seq2SeqTrainer integration
- MLflow experiment tracking
- Auto-detection of GPU/CPU with FP16 support
- Deterministic training (reproducible results)
- Best and last model checkpointing
- training_report.yaml generation

#### Stage 5 - Model Evaluation
- ROUGE metrics computation (ROUGE-1, ROUGE-2, ROUGE-L)
- Compression ratio analysis
- Batch inference on test set
- evaluation_report.yaml with sample predictions
- 4-beam search for quality summaries

#### Stage 6 - HuggingFace Hub Integration
- Automatic model pushing to HuggingFace Hub
- Auto-generated comprehensive model card
- Environment-based authentication (HF_TOKEN)
- Repository auto-creation
- Best model + tokenizer upload

#### Stage 7 - FastAPI Application
- REST API with /summarize endpoint
- Request/Response models with Pydantic
- Latency tracking (milliseconds)
- Health check endpoint
- Auto-loading from local model or HuggingFace Hub
- Interactive API documentation (Swagger)
- Model metadata in responses

#### Stage 8 - Docker Support
- Multi-stage Dockerfile for optimized images
- Health check integration
- Environment variable support (no baked secrets)
- .dockerignore for minimal image size
- CPU and GPU support

#### Stage 9 - CI/CD Pipeline
- GitHub Actions workflow
- Automated linting with Ruff
- Automated testing with pytest
- Docker build and push to Amazon ECR
- AWS deployment scaffold (ECS/EC2)
- Branch-based triggers (main, develop)

### Infrastructure
- Complete project structure following template.py
- Modular pipeline architecture (components + pipelines)
- Centralized configuration management
- Structured logging to file and stdout
- YAML-based configuration and parameters
- Comprehensive type hints with frozen dataclasses

### Testing
- Basic test suite for pipeline validation
- Config file validation tests
- Import tests
- pytest integration

### Documentation
- Comprehensive README.md
- API usage examples
- Docker deployment guide
- Stage-by-stage pipeline documentation
- Model card template
- Refactoring report
- Assumptions documentation

### Changed
- Migrated from ZIP-based data ingestion to HuggingFace Datasets
- Updated config.yaml to support all 9 pipeline stages
- Enhanced main.py with stage-specific execution
- Improved error handling and logging throughout

### Removed
- Temporary research notebooks (01_data_ingestion.ipynb, 02_data_transforamtion.ipynb, 03_model_trainer.ipynb)
- Old README content
- Hardcoded file paths and credentials

### Security
- All secrets moved to .env file
- .env in .gitignore by default
- No tokens logged or printed
- Fail-fast on missing credentials
- Docker secrets passed at runtime, not baked in

## [0.1.0] - Initial Development

### Added
- Basic project scaffolding
- Initial research notebooks
- Preliminary data ingestion from ZIP files

---

## Commit History

### Phase 0
- `chore: project structure refactor and cleanup`

### Stage 1
- `feat(stage1): implement SAMSum data ingestion + manifest artifacts`

### Stage 2
- `feat(stage2): implement data validation + validation report artifacts`

### Stage 3
- `feat(stage3): implement SAMSum transformation + PEGASUS tokenization artifacts`

### Stages 4-9
- All remaining stages implemented with complete functionality

---

**Note**: Each stage was implemented as an atomic commit following conventional commits format.
