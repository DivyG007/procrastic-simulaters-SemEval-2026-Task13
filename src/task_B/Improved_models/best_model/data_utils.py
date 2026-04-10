"""Data loading, preprocessing, and dataset utilities for Task B."""

import random
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from sklearn.utils.class_weight import compute_class_weight

from config import Config, NUM_LABELS


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_data_paths(cfg: Config) -> Tuple[str, str, Optional[str]]:
    """Download Task B data from HuggingFace if local parquet files are absent."""
    train_exists = _file_exists(cfg.train_path)
    val_exists = _file_exists(cfg.val_path)

    if train_exists and val_exists:
        test_path = cfg.test_path if _file_exists(cfg.test_path) else None
        return cfg.train_path, cfg.val_path, test_path

    print("Downloading SemEval-2026-Task13 Subtask B from HuggingFace...")
    hf_dataset = load_dataset("DaniilOr/SemEval-2026-Task13", "B")
    hf_dataset["train"].to_parquet("task_b_train.parquet")
    hf_dataset["validation"].to_parquet("task_b_val.parquet")

    test_path = None
    if "test" in hf_dataset:
        hf_dataset["test"].to_parquet("task_b_test.parquet")
        test_path = "task_b_test.parquet"

    cfg.train_path = "task_b_train.parquet"
    cfg.val_path = "task_b_val.parquet"
    cfg.test_path = test_path

    print(f"Train: {cfg.train_path}")
    print(f"Val:   {cfg.val_path}")
    print(f"Test:  {cfg.test_path}")

    return cfg.train_path, cfg.val_path, cfg.test_path


def stratified_subsample(df: pd.DataFrame, human_subset_size: int, random_state: int) -> pd.DataFrame:
    """Keep all minority classes and downsample only the human class."""
    human_df = df[df["label"] == 0]
    minority_df = df[df["label"] != 0]
    original_total = len(df)

    if len(human_df) > human_subset_size:
        human_df = human_df.sample(n=human_subset_size, random_state=random_state)

    out = pd.concat([human_df, minority_df], ignore_index=True)
    out = out.sample(frac=1, random_state=random_state).reset_index(drop=True)

    pct = len(out) / max(1, original_total) * 100
    print(f"Subsampled: {original_total} -> {len(out)} ({pct:.1f}%)")
    return out


def compute_balanced_weights(labels: np.ndarray, max_class_weight: float) -> torch.FloatTensor:
    """Compute sklearn balanced class weights and clip to a maximum value."""
    classes = np.arange(NUM_LABELS)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    weights = np.clip(weights, 1.0, max_class_weight)
    return torch.FloatTensor(weights)


def load_train_val_data(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, torch.FloatTensor]:
    """Load train/validation data, apply subsampling, and compute class weights."""
    train_df = pd.read_parquet(cfg.train_path)
    val_df = pd.read_parquet(cfg.val_path)

    if "code" not in train_df.columns or "label" not in train_df.columns:
        raise ValueError("Training data must contain 'code' and 'label' columns.")

    train_df = train_df.dropna(subset=["code", "label"]).copy()
    val_df = val_df.dropna(subset=["code", "label"]).copy()

    train_df["label"] = train_df["label"].astype(int)
    val_df["label"] = val_df["label"].astype(int)

    if cfg.use_subset:
        train_df = stratified_subsample(train_df, cfg.human_subset_size, cfg.random_seed)
        val_df = (
            val_df.groupby("label", group_keys=False)
            .apply(lambda x: x.sample(frac=cfg.val_fraction, random_state=cfg.random_seed))
            .reset_index(drop=True)
        )

    class_weights = compute_balanced_weights(train_df["label"].values, cfg.max_class_weight)
    return train_df, val_df, class_weights


def tokenize_datasets(train_df: pd.DataFrame, val_df: pd.DataFrame, tokenizer, max_length: int):
    """Build HuggingFace datasets and tokenize code snippets."""

    def tokenize_fn(examples):
        return tokenizer(
            examples["code"],
            truncation=True,
            padding=True,
            max_length=max_length,
        )

    train_dataset = Dataset.from_pandas(train_df[["code", "label"]])
    val_dataset = Dataset.from_pandas(val_df[["code", "label"]])

    train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=["code"])
    val_dataset = val_dataset.map(tokenize_fn, batched=True, remove_columns=["code"])

    train_dataset = train_dataset.rename_column("label", "labels")
    val_dataset = val_dataset.rename_column("label", "labels")

    return train_dataset, val_dataset


def _file_exists(path: Optional[str]) -> bool:
    """Return True when a non-empty file path exists."""
    if not path:
        return False
    try:
        with open(path, "rb"):
            return True
    except OSError:
        return False
