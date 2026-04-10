"""Configuration values for Task C baseline pipeline."""

from dataclasses import dataclass


@dataclass
class Config:
    """Notebook-equivalent default configuration."""

    # Data paths (kept same as baseline notebook)
    train_path: str = "/kaggle/input/datasets/vortex07/kaggle/task_c_training_set_1.parquet"
    val_path: str = "/kaggle/input/datasets/vortex07/kaggle/task_c_validation_set.parquet"
    test_path: str = "/kaggle/input/datasets/vortex07/kaggle/task_c_test_set_sample.parquet"

    # Model and tokenizer
    model_name: str = "microsoft/codebert-base"
    max_length: int = 128

    # Sampling and training
    sample_fraction: float = 0.1
    random_seed: int = 42
    output_dir: str = "taskC-model"
    num_epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 2e-5

    # Inference
    test_max_length: int = 256
    test_batch_size: int = 32
    submission_path: str = "/kaggle/working/submission.csv"


def default_config() -> Config:
    """Return default Task C baseline config."""
    return Config()
