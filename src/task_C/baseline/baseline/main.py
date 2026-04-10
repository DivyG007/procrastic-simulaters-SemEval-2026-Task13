"""Single-entry script for Task C baseline modular pipeline.

Run:
    python3 main.py
"""

import logging
import os
import warnings

import torch

from config import default_config
from predict import predict_with_trainer
from trainer import CodeBERTTrainer


os.environ["WANDB_DISABLED"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=''"
logging.disable(logging.WARNING)
warnings.filterwarnings("ignore")


def run() -> None:
    """Run train/eval and optional test prediction generation."""
    cfg = default_config()

    trainer_obj = CodeBERTTrainer(cfg)
    trainer = trainer_obj.run_full_pipeline()

    if cfg.test_path:
        predict_with_trainer(
            trainer_obj=trainer_obj,
            parquet_path=cfg.test_path,
            output_path=cfg.submission_path,
            max_length=cfg.test_max_length,
            batch_size=cfg.test_batch_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        print(f"Wrote: {cfg.submission_path}")


if __name__ == "__main__":
    run()
