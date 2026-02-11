from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    dataset_name: str
    raw_data_path: Path

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    data_path: Path

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    tokenizer_name: str

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    data_path: Path
    model_ckpt: str
    model_best_path: Path
    model_last_path: Path
    num_train_epochs: int
    learning_rate: float
    warmup_steps: int
    weight_decay: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    fp16: bool
    eval_strategy: str
    eval_steps: int
    logging_steps: int
    predict_with_generate: bool
    generation_max_length: int
    save_strategy: str
    save_steps: int

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    model_path: Path
    data_path: Path
    tokenizer_name: str

@dataclass(frozen=True)
class ModelPusherConfig:
    model_path: Path
    tokenizer_name: str
    hf_token: str
    hf_repo_id: str