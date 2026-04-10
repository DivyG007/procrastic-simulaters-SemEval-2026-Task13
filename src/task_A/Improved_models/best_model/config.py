"""
Configuration and hyperparameters for the CodeBERT Deep Head model.
"""

import os

# --- Model Configuration ---
MODEL_NAME = "microsoft/codebert-base"
MAX_LENGTH = 512
NUM_LABELS = 2
NUM_CODE_FEATURES = 8

# --- Training Hyperparameters ---
SAMPLE_SIZE = 40_000
RANDOM_SEED = 42
DROPOUT = 0.1
NUM_EPOCHS = 10
BATCH_SIZE = 16
BASE_LR = 2e-5
HEAD_LR = 1e-3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
GRAD_CLIP = 1.0
LLRD_FACTOR = 0.95

# --- Output and Logs ---
OUTPUT_DIR = "./codebert_deep_head_results"
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")

# --- Data Paths ---
# Original Kaggle path: /kaggle/input/datasets/daniilor/semeval-2026-task13/SemEval-2026-Task13/task_a/task_a_training_set_1.parquet
TRAIN_PARQUET = "/kaggle/input/datasets/daniilor/semeval-2026-task13/SemEval-2026-Task13/task_a/task_a_training_set_1.parquet"
TEST_PARQUET = "/kaggle/input/datasets/daniilor/semeval-2026-task13/SemEval-2026-Task13/task_a/task_a_test_set_sample.parquet"

# --- Environment ---
WANDB_DISABLED = "true"
