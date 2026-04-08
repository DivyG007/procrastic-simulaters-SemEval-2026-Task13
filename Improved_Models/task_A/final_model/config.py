"""
config.py — Shared Constants & Hyperparameters
===============================================
Central configuration for all Task A training scripts and the ensemble
pipeline.  Kaggle dataset paths, model registry, training hyperparams,
and evaluation category sets live here so every script stays in sync.
"""

import os

# ── Random seed (reproducibility) ──────────────────────────────────────
RANDOM_SEED = 42

# ── Data paths (Kaggle dataset layout) ─────────────────────────────────
DATA_ROOT = (
    "/kaggle/input/datasets/daniilor/semeval-2026-task13/"
    "SemEval-2026-Task13/task_a"
)

TRAIN_PARQUET = os.path.join(DATA_ROOT, "task_a_training_set_1.parquet")
TEST_PARQUET  = os.path.join(DATA_ROOT, "task_a_test_set_sample.parquet")

# ── Sampling & splits ──────────────────────────────────────────────────
SAMPLE_SIZE = 40_000          # stratified subsample from ~100 K rows
TEST_RATIO  = 0.20            # 80 / 10 / 10  (train / val / test)
VAL_TEST_RATIO = 0.50         # half of the 20 % goes to val, half to test

# ── Tokeniser ──────────────────────────────────────────────────────────
MAX_LENGTH = 512

# ── Training hyperparameters ───────────────────────────────────────────
NUM_EPOCHS   = 10
BATCH_SIZE   = 16
BASE_LR      = 2e-5
HEAD_LR      = 1e-3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
GRAD_CLIP    = 1.0
LLRD_FACTOR  = 0.95
DROPOUT      = 0.1

# ── Stylometric features ──────────────────────────────────────────────
NUM_CODE_FEATURES = 8

# ── Model registry ─────────────────────────────────────────────────────
#    Maps a short tag → HuggingFace model name.
MODEL_REGISTRY = {
    "codebert":      "microsoft/codebert-base",
    "graphcodebert": "microsoft/graphcodebert-base",
    "unixcoder":     "microsoft/unixcoder-base",
}

# ── Output directories (on Kaggle /kaggle/working/) ────────────────────
OUTPUT_ROOT = "/kaggle/working"

def output_dir_for(tag: str) -> str:
    """Return the checkpoint / output directory for a given model tag."""
    return os.path.join(OUTPUT_ROOT, f"{tag}_deep_head")

ENSEMBLE_CACHE_DIR = os.path.join(OUTPUT_ROOT, "ensemble_cache")
ENSEMBLE_CSV       = os.path.join(OUTPUT_ROOT, "ensemble_submission.csv")

# ── Evaluation category sets ───────────────────────────────────────────
SEEN_LANGS     = {"c++", "cpp", "python", "java"}
UNSEEN_LANGS   = {"go", "php", "c#", "csharp", "c", "javascript", "js"}
SEEN_DOMAINS   = {"algorithmic"}
UNSEEN_DOMAINS = {"research", "production"}

# ── Label mapping ──────────────────────────────────────────────────────
LABEL_MAP = {0: "human", 1: "machine"}
NUM_LABELS = 2
