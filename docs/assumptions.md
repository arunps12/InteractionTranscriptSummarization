# Assumptions Documentation

**Project:** Interaction Transcript Summarization  
**Date:** 2026-02-11  
**Phase:** 0 - Structure Audit & Refactoring

---

## General Assumptions

### 1. Dataset Source
**Assumption:** SAMSum dataset will be loaded directly from HuggingFace Datasets library (`datasets.load_dataset("samsum")`).

**Rationale:** 
- Specification states "Load SAMSum via HuggingFace Datasets"
- More reliable than downloading zips from GitHub
- Eliminates need for manual data hosting
- Standard practice in modern ML pipelines

**Previous Approach (Removed):** ZIP file download from GitHub URL

---

### 2. Artifact Directory Structure
**Assumption:** Artifacts will use timestamped directories only for data ingestion, validation stages. Model training artifacts will use static paths for easier reference.

**Rationale:**
- Specification mentions "Timestamped artifact directories"
- But also specifies fixed paths like `artifacts/model_trainer/model_best`
- Compromise: Use timestamps for data stages, fixed for model artifacts
- Easier for deployment and model loading

**Implementation:**
```
artifacts/
├── <timestamp>/data_ingestion/     # Timestamped
├── <timestamp>/data_validation/    # Timestamped
├── data_transformation/            # Static (referenced by trainer)
├── model_trainer/                  # Static
│   ├── model_best/                 # Fixed path for deployment
│   └── model_last/                 # Fixed path for checkpointing
└── model_evaluation/               # Static
```

---

### 3. Environment Variables
**Assumption:** All secrets will be stored in `.env` file in project root and loaded via `python-dotenv`.

**Required Variables:**
```bash
HF_TOKEN=<huggingface_token>
HF_REPO_ID=arunps12/pegasus-samsum-dialogue-summarizer
```

**Rationale:**
- Specification explicitly requires .env usage
- .env already in .gitignore
- Standard practice for secret management
- Fails fast if secrets missing (security best practice)

---

### 4. Training Strategy
**Assumption:** Training will prioritize `model_best` based on validation metrics over `model_last`.

**Implementation:**
- `save_strategy: "steps"` with large `save_steps` for model_last
- `evaluation_strategy: "steps"` with smaller `eval_steps`
- Best model auto-saved by HuggingFace Trainer based on ROUGE metrics
- Only `model_best` pushed to HuggingFace Hub (per specification)

**Rationale:** Specification states "Push ONLY model_best + tokenizer"

---

### 5. MLflow Integration
**Assumption:** MLflow tracking will log to local `mlruns/` directory by default, with option to use remote tracking server via environment variable.

**Environment Variables (Optional):**
```bash
MLFLOW_TRACKING_URI=<remote_uri>  # Optional, defaults to local
MLFLOW_EXPERIMENT_NAME=pegasus-samsum-training  # Optional
```

**Rationale:**
- Local logging works out-of-box for development
- Production can point to remote server
- Specification mentions "Log with MLflow"

---

### 6. Data Validation Criteria
**Assumption:** Validation checks will include:
1. Required columns exist (`dialogue`, `summary`)
2. No null/empty dialogues or summaries
3. Minimum dialogue length: 10 characters
4. Minimum summary length: 5 characters
5. Maximum dialogue length: 10,000 characters (sanity check)
6. Summary should be shorter than dialogue (compression check)

**Rationale:** Specification says "validate schema, remove empty samples, length sanity checks"

---

### 7. Tokenization Parameters
**Assumption:** 
- Max input length: 1024 tokens (PEGASUS limit)
- Max target length: 128 tokens (summary length)
- Truncation enabled for longer inputs

**Rationale:**
- PEGASUS model architecture constraints
- SAMSum dialogues typically fit within 1024 tokens
- Summaries are concise (128 sufficient)

---

### 8. Deterministic Training
**Assumption:** Random seeds will be set for reproducibility:
- Python random seed: 42
- NumPy random seed: 42
- PyTorch manual seed: 42
- CUDA deterministic mode: enabled

**Implementation:** Will be set in Model Trainer component

**Rationale:** Specification requires "Deterministic training" and "Deterministic seeds"

---

### 9. ROUGE Metrics
**Assumption:** Evaluation will compute:
- ROUGE-1 (unigram overlap)
- ROUGE-2 (bigram overlap)
- ROUGE-L (longest common subsequence)
- Compression ratio (summary_len / dialogue_len)

**Rationale:** Specification explicitly lists these metrics

---

### 10. FastAPI Endpoint Design
**Assumption:** Single `/summarize` endpoint with the following contract:

**Request:**
```json
{
  "text": "Hannah: Hey, do you have Betty's number?\nAmanda: Lemme check\nHannah: <file_gif>\nAmanda: Sorry, can't find it."
}
```

**Response:**
```json
{
  "summary": "Hannah asked Amanda for Betty's number but Amanda couldn't find it.",
  "model": "google/pegasus-cnn_dailymail",
  "version": "1.0.0",
  "latency_ms": 245.3
}
```

**Rationale:** Specification defines this exact structure

---

### 11. Docker Configuration
**Assumption:** 
- Base image: `python:3.10-slim` (smaller footprint)
- Multi-stage build not required (model size manageable)
- Port: 8000 (FastAPI default)
- Healthcheck: `GET /health` endpoint
- Environment variables passed at runtime (not baked in)

**Rationale:** Specification says ".env NOT baked into image" and "CPU + optional GPU support"

---

### 12. GPU Support
**Assumption:** Training will auto-detect GPU availability and use FP16 mixed precision if available.

**Implementation:**
```python
import torch
fp16 = torch.cuda.is_available()  # Auto-detect
```

**Rationale:** Specification says "FP16 if GPU available"

---

### 13. AWS Deployment
**Assumption:** 
- Primary deployment: Amazon ECS Fargate (serverless container)
- Alternative: EC2 with Docker (simpler, scaffold acceptable per spec)
- Container registry: Amazon ECR
- Model storage: Primarily HuggingFace Hub (spec states "Artifact storage via HF Hub (primary)")

**Rationale:** Specification allows "scaffold acceptable" and prioritizes HF Hub

---

### 14. Testing Strategy
**Assumption:** 
- Unit tests for each component (pytest)
- Integration test for end-to-end pipeline
- API tests for FastAPI endpoints
- Mock external dependencies (HuggingFace API, MLflow)

**Rationale:** Specification requires "pytest must pass"

---

### 15. Linting and Code Quality
**Assumption:** 
- Linter: Ruff (modern, fast Python linter)
- Formatter: Ruff format
- Type checking: Basic (via dataclasses, no mypy required)

**Rationale:** Specification requires "Ruff lint must pass"

---

### 16. Git Workflow
**Assumption:** 
- Each pipeline stage = 1 atomic commit
- Commit messages follow conventional commits format
- No mixed changes in single commit
- All commits must pass tests before pushing

**Rationale:** Specification explicitly requires atomic commits per stage

---

### 17. Model Card Generation
**Assumption:** HuggingFace Hub push will auto-generate a model card (README.md) with:
- Model description
- Training dataset (SAMSum)
- Metrics (ROUGE scores)
- Usage example
- License (MIT 2.0)

**Rationale:** Specification mentions "Generate model card README"

---

### 18. Logging Strategy
**Assumption:** 
- Log level: INFO for normal operation
- Log format: `[timestamp: level: module: message]`
- Log destination: Both file (`logs/continuos_logs.log`) and stdout
- No sensitive data (tokens, keys) logged

**Rationale:** Existing logging configuration + security best practices

---

### 19. Error Handling
**Assumption:** 
- Fail fast on missing configuration
- Fail fast on missing secrets
- Graceful degradation NOT used (explicit errors better for ML pipelines)
- All errors logged before raising

**Rationale:** ML pipelines should fail loudly to prevent silent data/model corruption

---

### 20. Version Control
**Assumption:** 
- Model version tracked in metadata files
- API version returned in response
- Semantic versioning (1.0.0 for initial release)

**Rationale:** Specification shows version in API response

---

## Implementation Priority

These assumptions will guide implementation in the following order:
1. ✅ Phase 0: Structure refactoring (current)
2. Phase 1: Stages 1-9 implementation
3. Phase 2: Testing and CI/CD
4. Phase 3: Documentation finalization

---

**Note:** These assumptions are documented as part of the autonomous implementation process. Any deviations discovered during implementation will be documented and adjusted accordingly.
