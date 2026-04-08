"""
train_codebert.py — CodeBERT Training Script
==============================================
Self-contained script to fine-tune CodeBERT (microsoft/codebert-base)
with a deep classification head for Task A (human vs. machine code).

Usage on Kaggle:
    python train_codebert.py

This script:
  1. Installs / upgrades required packages
  2. Loads & cleans the training parquet
  3. Builds HuggingFace datasets with stylometric features
  4. Trains DeepHeadModel(codebert) with LLRD + cosine schedule + fp16
  5. Evaluates on the held-out test split (4-category breakdown)
  6. Saves the model checkpoint and probability .npy for ensemble use
"""

import os
import sys

# ── Ensure local imports work when run as a script ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ["WANDB_DISABLED"] = "true"

import numpy as np
import torch
import warnings
from transformers import RobertaTokenizer, EarlyStoppingCallback

from config import (
    MODEL_REGISTRY,
    NUM_LABELS,
    NUM_CODE_FEATURES,
    DROPOUT,
    NUM_EPOCHS,
    BATCH_SIZE,
    BASE_LR,
    HEAD_LR,
    WEIGHT_DECAY,
    WARMUP_RATIO,
    GRAD_CLIP,
    LLRD_FACTOR,
    MAX_LENGTH,
    TEST_PARQUET,
    ENSEMBLE_CACHE_DIR,
    output_dir_for,
)
from data_utils import load_and_prepare_data, make_hf_dataset
from model import DeepHeadModel
from train_utils import (
    FeaturesDataCollator,
    DeepHeadTrainer,
    compute_metrics,
    evaluate_by_category,
    predict_on_dataset,
    save_probabilities,
    cleanup_gpu,
)
from transformers import TrainingArguments

warnings.filterwarnings("ignore")

# ── Model tag & backbone ───────────────────────────────────────────────
TAG        = "codebert"
MODEL_NAME = MODEL_REGISTRY[TAG]
OUTPUT_DIR = output_dir_for(TAG)


def main():
    """Full training + evaluation pipeline for CodeBERT."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'=' * 70}")
    print(f"  Training: {TAG}  ({MODEL_NAME})")
    print(f"  Device: {device}")
    print(f"{'=' * 70}\n")

    # ── 1. Data Loading & Cleaning ──
    train_df, val_df, test_df = load_and_prepare_data()

    # ── 2. Tokeniser & HF Datasets ──
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = make_hf_dataset(train_df, tokenizer, max_length=MAX_LENGTH)
    val_dataset   = make_hf_dataset(val_df,   tokenizer, max_length=MAX_LENGTH)

    print(f"HF datasets ready — Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # ── 3. Model ──
    model = DeepHeadModel(
        model_name=MODEL_NAME,
        num_labels=NUM_LABELS,
        num_features=NUM_CODE_FEATURES,
        dropout=DROPOUT,
    ).to(device)

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {total_p:,} total, {train_p:,} trainable")

    # ── 4. Training Arguments ──
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        weight_decay=WEIGHT_DECAY,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        remove_unused_columns=False,
        learning_rate=BASE_LR,
        lr_scheduler_type="cosine",          # overridden by custom scheduler
        warmup_ratio=WARMUP_RATIO,
        max_grad_norm=GRAD_CLIP,
        save_total_limit=2,
        report_to=[],
        fp16=True,
    )

    data_collator = FeaturesDataCollator(tokenizer=tokenizer)

    trainer = DeepHeadTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        llrd_factor=LLRD_FACTOR,
        head_lr=HEAD_LR,
    )

    # ── 5. Train ──
    print("Starting training ...")
    trainer.train()

    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Training complete. Best model saved to {OUTPUT_DIR}")

    # ── 6. Evaluate on held-out test split ──
    print(f"\nEvaluating on held-out test set ({len(test_df)} samples) ...")
    test_results, test_probs = predict_on_dataset(
        model, tokenizer, test_df,
        max_length=MAX_LENGTH, batch_size=32, device=device,
    )
    evaluate_by_category(test_results, tag=f"{TAG}-DeepHead")

    # Save probabilities for ensemble
    test_ids = np.arange(len(test_df)).astype(str)
    save_probabilities(test_probs, model_tag=TAG, output_dir=ENSEMBLE_CACHE_DIR, ids=test_ids)

    # ── 7. Official test parquet (if available) ──
    if os.path.exists(TEST_PARQUET):
        import pandas as pd
        print(f"\nEvaluating on official test set: {TEST_PARQUET}")
        official_test_df = pd.read_parquet(TEST_PARQUET)
        official_test_df = official_test_df.dropna(subset=["code"]).reset_index(drop=True)
        if "label" in official_test_df.columns:
            official_test_df["label"] = official_test_df["label"].astype(int)

        print(f"Official test set size: {len(official_test_df)}")
        official_results, official_probs = predict_on_dataset(
            model, tokenizer, official_test_df,
            max_length=MAX_LENGTH, batch_size=32, device=device,
        )
        evaluate_by_category(official_results, tag=f"{TAG}-DeepHead (Official Test)")

        # Overwrite cache with official test probs for ensemble
        official_ids = (
            official_test_df["ID"].astype(str).values
            if "ID" in official_test_df.columns
            else np.arange(len(official_test_df)).astype(str)
        )
        save_probabilities(official_probs, model_tag=TAG, output_dir=ENSEMBLE_CACHE_DIR, ids=official_ids)
    else:
        print(f"Official test parquet not found at {TEST_PARQUET}, skipping.")

    # ── 8. Summary ──
    print(f"\n{'=' * 70}")
    print(f"  {TAG.upper()} CONFIGURATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Model:             {MODEL_NAME}")
    print(f"  Head:              776 → 256 → 128 → 64 → 2  (GELU + LayerNorm)")
    print(f"  Features:          {NUM_CODE_FEATURES} handcrafted stylometric")
    print(f"  Epochs:            {NUM_EPOCHS}")
    print(f"  Batch size:        {BATCH_SIZE}")
    print(f"  Base LR:           {BASE_LR}")
    print(f"  Head LR:           {HEAD_LR}")
    print(f"  LLRD factor:       {LLRD_FACTOR}")
    print(f"  Dropout:           {DROPOUT}")
    print(f"  Precision:         fp16 (mixed)")
    print(f"  Early stopping:    patience=3, metric=f1")
    print(f"{'=' * 70}")

    # Free GPU memory for the next model
    del model, trainer
    cleanup_gpu()

    return OUTPUT_DIR


if __name__ == "__main__":
    main()
