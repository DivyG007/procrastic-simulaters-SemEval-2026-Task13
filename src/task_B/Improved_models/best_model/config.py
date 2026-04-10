"""Configuration for Task B GraphCodeBERT best model pipeline."""

from dataclasses import dataclass
from typing import Dict, Optional


ID_TO_LABEL: Dict[int, str] = {
    0: "human",
    1: "deepseek",
    2: "qwen",
    3: "01-ai",
    4: "bigcode",
    5: "gemma",
    6: "phi",
    7: "meta-llama",
    8: "ibm-granite",
    9: "mistral",
    10: "openai",
}
LABEL_TO_ID: Dict[str, int] = {v: k for k, v in ID_TO_LABEL.items()}
NUM_LABELS = len(ID_TO_LABEL)


@dataclass
class Config:
    """Experiment configuration with notebook-equivalent defaults."""

    # Model
    model_name: str = "microsoft/graphcodebert-base"

    # Data paths
    train_path: str = "task_b_train.parquet"
    val_path: str = "task_b_val.parquet"
    test_path: Optional[str] = "task_b_test.parquet"

    # Subsampling
    use_subset: bool = True
    human_subset_size: int = 45000
    val_fraction: float = 0.1
    random_seed: int = 42

    # Training hyperparameters
    max_length: int = 512
    batch_size: int = 8
    grad_accum_steps: int = 4
    learning_rate: float = 3e-5
    num_epochs: int = 6
    weight_decay: float = 0.01
    warmup_steps: int = 500
    label_smoothing: float = 0.0
    early_stopping_patience: int = 2

    # Losses and optimization
    focal_gamma: float = 2.0
    supcon_weight: float = 0.2
    supcon_temperature: float = 0.07
    llrd_decay: float = 0.9
    mixup_alpha: float = 0.4
    max_class_weight: float = 8.0

    # Output
    output_dir: str = "/kaggle/working/results_graphcodebert_taskB"
    submission_csv: str = "submission_b_graphcodebert.csv"

    # Runtime
    fp16: bool = True


def default_config() -> Config:
    """Return default configuration."""
    return Config()
