"""
Main entry point for training and evaluating the CodeBERT Deep Head model.
"""

import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, TrainingArguments, EarlyStoppingCallback
import warnings

# Add current directory to path to allow script-style imports.
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *
from data_utils import clean_task_a, make_hf_dataset, FeaturesDataCollator
from model import DeepHeadCodeBERT
from trainer import DeepHeadTrainer
from eval import compute_metrics, predict_on_dataset, evaluate_by_category

warnings.filterwarnings("ignore")


def main():
    """Main execution flow."""
    os.environ["WANDB_DISABLED"] = WANDB_DISABLED
    print("Initializing CodeBERT-DeepHead Pipeline...")

    # 1. Load Data
    if not os.path.exists(TRAIN_PARQUET):
        print(f"ERROR: Training data not found at {TRAIN_PARQUET}")
        print("Please update TRAIN_PARQUET in config.py with the correct path.")
        return

    print(f"Loading data from {TRAIN_PARQUET}...")
    raw_df = pd.read_parquet(TRAIN_PARQUET)
    raw_df["label"] = raw_df["label"].astype(int)

    # 2. Clean Data
    print("Cleaning dataset...")
    df_clean = clean_task_a(raw_df)

    # 3. Stratified Sampling
    if SAMPLE_SIZE < len(df_clean):
        print(f"Sampling {SAMPLE_SIZE} rows stratified by label...")
        df_sampled = df_clean.groupby("label", group_keys=False).apply(
            lambda x: x.sample(
                n=max(1, int(SAMPLE_SIZE * len(x) / len(df_clean))),
                random_state=RANDOM_SEED,
            )
        ).reset_index(drop=True)
    else:
        df_sampled = df_clean.copy()

    # 4. Train/Val/Test Split (80/10/10)
    train_df, temp_df = train_test_split(
        df_sampled,
        test_size=0.20,
        stratify=df_sampled["label"],
        random_state=RANDOM_SEED,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=RANDOM_SEED,
    )
    print(f"Splits — Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 5. Prepare HF Datasets
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = make_hf_dataset(train_df, tokenizer, MAX_LENGTH)
    val_dataset = make_hf_dataset(val_df, tokenizer, MAX_LENGTH)

    # 6. Initialize Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = DeepHeadCodeBERT(
        model_name=MODEL_NAME,
        num_labels=NUM_LABELS,
        num_features=NUM_CODE_FEATURES,
        dropout=DROPOUT,
    ).to(device)

    # 7. Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        weight_decay=WEIGHT_DECAY,
        logging_dir=LOGS_DIR,
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
        warmup_ratio=WARMUP_RATIO,
        max_grad_norm=GRAD_CLIP,
        save_total_limit=2,
        report_to=[],
        fp16=torch.cuda.is_available(),
    )

    data_collator = FeaturesDataCollator(tokenizer=tokenizer)

    # 8. Initialize Trainer
    trainer = DeepHeadTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        llrd_factor=LLRD_FACTOR,
        head_lr=HEAD_LR,
    )

    # 9. Train
    print("Starting training...")
    trainer.train()

    # 10. Save Model
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model and tokenizer saved to {OUTPUT_DIR}")

    # 11. Evaluation
    print("Evaluating on held-out test split...")
    test_results = predict_on_dataset(
        model,
        tokenizer,
        test_df,
        max_length=MAX_LENGTH,
        batch_size=32,
        device=device,
    )
    evaluate_by_category(test_results, tag="CodeBERT-DeepHead (Held-out)")

    # 12. Evaluate on official test set if available
    if os.path.exists(TEST_PARQUET):
        print(f"\nEvaluating on official test set: {TEST_PARQUET}")
        off_df = pd.read_parquet(TEST_PARQUET)
        off_df = off_df.dropna(subset=["code"]).reset_index(drop=True)
        if "label" in off_df.columns:
            off_df["label"] = off_df["label"].astype(int)

        off_results = predict_on_dataset(
            model,
            tokenizer,
            off_df,
            max_length=MAX_LENGTH,
            batch_size=32,
            device=device,
        )
        evaluate_by_category(off_results, tag="CodeBERT-DeepHead (Official Test)")

    print("\nPipeline execution complete.")


if __name__ == "__main__":
    main()
