"""Single-entry script for Task B improved best-model pipeline.

Run:
    python3 main.py
"""

import gc
import os

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import RobertaTokenizer

from calibration import calibrate_thresholds, predict_with_thresholds
from config import ID_TO_LABEL, NUM_LABELS, default_config
from data_utils import load_train_val_data, resolve_data_paths, set_seed, tokenize_datasets
from predict import predict_with_trainer
from trainer import GraphCodeBERTTrainerB


os.environ["WANDB_DISABLED"] = "true"


def run() -> None:
    """Execute full notebook-equivalent pipeline as modular Python code."""
    cfg = default_config()
    set_seed(cfg.random_seed)

    resolve_data_paths(cfg)

    print(f"Model: {cfg.model_name}")
    print(f"Task B labels: {list(ID_TO_LABEL.values())}")

    train_df, val_df, class_weights = load_train_val_data(cfg)

    trainer_obj = GraphCodeBERTTrainerB(cfg)
    trainer_obj.class_weights = class_weights

    # Initialize tokenizer once for dataset tokenization.
    trainer_obj.tokenizer = RobertaTokenizer.from_pretrained(cfg.model_name)
    train_dataset, val_dataset = tokenize_datasets(train_df, val_df, trainer_obj.tokenizer, cfg.max_length)

    # Train + evaluate.
    artifacts = trainer_obj.run_full_pipeline(train_dataset, val_dataset)

    # Threshold calibration.
    val_predictions = artifacts.trainer.predict(artifacts.val_dataset)
    val_logits = val_predictions.predictions
    val_labels = val_predictions.label_ids

    thresholds = calibrate_thresholds(val_logits, val_labels, NUM_LABELS)
    calibrated_preds = predict_with_thresholds(val_logits, thresholds)

    calibrated_f1 = f1_score(val_labels, calibrated_preds, average="macro", zero_division=0)
    uncalibrated_preds = np.argmax(val_logits, axis=1)
    uncalibrated_f1 = f1_score(val_labels, uncalibrated_preds, average="macro", zero_division=0)

    target_names = [ID_TO_LABEL[i] for i in range(NUM_LABELS)]
    print("\nCalibrated report:")
    print(classification_report(val_labels, calibrated_preds, target_names=target_names, zero_division=0))
    print("Calibrated confusion matrix:")
    print(confusion_matrix(val_labels, calibrated_preds, labels=range(NUM_LABELS)))
    print(f"Uncalibrated Macro F1: {uncalibrated_f1:.4f}")
    print(f"Calibrated Macro F1:   {calibrated_f1:.4f}")
    print(f"Improvement: {(calibrated_f1 - uncalibrated_f1) * 100:.2f} pts")

    # Test-set prediction.
    if cfg.test_path:
        predict_with_trainer(
            trainer_obj=trainer_obj,
            parquet_path=cfg.test_path,
            output_path=cfg.submission_csv,
            max_length=cfg.max_length,
            batch_size=32,
            device="cuda" if torch.cuda.is_available() else "cpu",
            thresholds=thresholds,
        )
        print(f"Wrote: {cfg.submission_csv}")
    else:
        print("No test split found. Skipping submission generation.")

    # Cleanup.
    del artifacts, trainer_obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run()
