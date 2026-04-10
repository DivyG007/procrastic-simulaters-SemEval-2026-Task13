"""Data loading and tokenization helpers for Task C baseline."""

import pandas as pd
from datasets import Dataset


def load_data(train_path: str, val_path: str, sample_fraction: float, random_seed: int):
    """Load train/val parquet files and apply notebook-style sampling."""
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    if "code" not in train_df.columns or "label" not in train_df.columns:
        raise ValueError("Dataset must contain 'code' and 'label' columns")

    train_df = train_df.dropna(subset=["code", "label"]).copy()
    val_df = val_df.dropna(subset=["code", "label"]).copy()

    train_df["label"] = train_df["label"].astype(int)
    val_df["label"] = val_df["label"].astype(int)

    train_df = train_df.sample(frac=sample_fraction, random_state=random_seed).reset_index(drop=True)
    val_df = val_df.sample(frac=sample_fraction, random_state=random_seed).reset_index(drop=True)

    return train_df, val_df


def tokenize_datasets(train_df, val_df, tokenizer, max_length: int):
    """Tokenize train/validation datasets exactly like notebook logic."""

    def tokenize_function(examples):
        return tokenizer(
            examples["code"],
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )

    train_dataset = Dataset.from_pandas(train_df[["code", "label"]])
    val_dataset = Dataset.from_pandas(val_df[["code", "label"]])

    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["code"])
    val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["code"])

    train_dataset = train_dataset.rename_column("label", "labels")
    val_dataset = val_dataset.rename_column("label", "labels")
    return train_dataset, val_dataset
