"""
Ensemble Pipeline: Soft Voting & Weighted Averaging
====================================================
Combines predictions from CodeBERT, GraphCodeBERT, and UniXcoder
for SemEval-2026 Task 13 (Subtasks A & B).

Strategies:
  1. Soft Voting     – average class probabilities, then argmax
  2. Weighted Average – weighted sum of class probabilities by per-model weights
  3. Per-Model Predict – generate per-model softmax outputs (cached as .npy)

Usage (CLI):
  # Generate per-model probabilities
  python ensemble_pipeline.py predict \
      --model_paths codebert_dir graphcodebert_dir unixcoder_dir \
      --parquet_path task_a_test.parquet \
      --output_dir ./ensemble_cache \
      --task A

  # Soft voting ensemble
  python ensemble_pipeline.py ensemble \
      --prob_dir ./ensemble_cache \
      --output_csv submission.csv \
      --strategy soft_vote

  # Weighted averaging ensemble
  python ensemble_pipeline.py ensemble \
      --prob_dir ./ensemble_cache \
      --output_csv submission.csv \
      --strategy weighted_avg \
      --weights 0.4 0.35 0.25

  # Full pipeline (predict + ensemble)
  python ensemble_pipeline.py full \
      --model_paths codebert_dir graphcodebert_dir unixcoder_dir \
      --parquet_path task_a_test.parquet \
      --output_dir ./ensemble_cache \
      --output_csv submission.csv \
      --task A \
      --strategy weighted_avg \
      --weights 0.4 0.35 0.25
"""

import os

# Suppress XLA / TensorFlow and CUDA plugin registration warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["USE_TF"] = "0"

import gc
import json
import argparse
import logging
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ensemble")

# ── Model registry ─────────────────────────────────────────────────────
MODEL_NAMES = [
    "microsoft/codebert-base",
    "microsoft/graphcodebert-base",
    "microsoft/unixcoder-base",
]

MODEL_TAGS = ["codebert", "graphcodebert", "unixcoder"]

# ── Label mappings ──────────────────────────────────────────────────────
TASK_LABELS = {
    "A": {0: "human", 1: "machine"},
    "B": {
        0: "human", 1: "deepseek", 2: "qwen", 3: "01-ai",
        4: "bigcode", 5: "gemma", 6: "phi", 7: "meta-llama",
        8: "ibm-granite", 9: "mistral", 10: "openai",
    },
}


# =====================================================================
#  Per-Model Inference
# =====================================================================

def load_model_and_tokenizer(model_path: str, device: str):
    """
    Load a fine-tuned RoBERTa-based model and its tokenizer.

    Supports standard RobertaForSequenceClassification checkpoints
    (from CodeBERT, GraphCodeBERT, or UniXcoder fine-tuning).
    """
    from transformers import RobertaTokenizer, RobertaForSequenceClassification

    logger.info(f"Loading model from: {model_path}")
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    num_labels = model.config.num_labels
    logger.info(f"  → {num_labels} labels | device={device}")
    return model, tokenizer


@torch.no_grad()
def predict_probabilities(
    model,
    tokenizer,
    codes: List[str],
    max_length: int = 512,
    batch_size: int = 32,
    device: str = "cuda",
) -> np.ndarray:
    """
    Run inference on a list of code strings.

    Returns:
        probs: np.ndarray of shape (N, num_labels) with softmax probabilities
    """
    all_probs = []

    for i in tqdm(range(0, len(codes), batch_size), desc="  Inference"):
        batch = codes[i : i + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)

    return np.concatenate(all_probs, axis=0)


def generate_model_predictions(
    model_path: str,
    parquet_path: str,
    output_dir: str,
    model_tag: str,
    max_length: int = 512,
    batch_size: int = 32,
    device: str = "cuda",
) -> Tuple[np.ndarray, List[str]]:
    """
    Run inference for a single model and cache the softmax probabilities.

    Saves:
      - {output_dir}/{model_tag}_probs.npy   (N x num_labels)
      - {output_dir}/sample_ids.npy           (N,) — shared across models

    Returns:
        (probs, ids)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    df = pd.read_parquet(parquet_path)
    if "code" not in df.columns:
        raise ValueError(f"Parquet '{parquet_path}' must contain a 'code' column")

    df = df.dropna(subset=["code"]).reset_index(drop=True)
    codes = df["code"].tolist()

    # Determine ID column
    if "ID" in df.columns:
        ids = df["ID"].astype(str).tolist()
    elif "id" in df.columns:
        ids = df["id"].astype(str).tolist()
    else:
        ids = [str(i) for i in range(len(df))]

    # Load model
    model, tokenizer = load_model_and_tokenizer(model_path, device)

    # Predict
    logger.info(f"[{model_tag}] Predicting on {len(codes)} samples …")
    probs = predict_probabilities(
        model, tokenizer, codes,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
    )

    # Cache
    prob_path = os.path.join(output_dir, f"{model_tag}_probs.npy")
    id_path = os.path.join(output_dir, "sample_ids.npy")
    np.save(prob_path, probs)
    np.save(id_path, np.array(ids))
    logger.info(f"[{model_tag}] Saved probabilities → {prob_path}  shape={probs.shape}")

    # Free GPU memory
    del model, tokenizer
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return probs, ids


# =====================================================================
#  Ensemble Strategies
# =====================================================================

def soft_vote(prob_list: List[np.ndarray]) -> np.ndarray:
    """
    Soft Voting: simple average of probability distributions.

    Each model gets equal weight. Final prediction = argmax of mean probs.

    Args:
        prob_list: list of K arrays, each (N, C) softmax probabilities

    Returns:
        averaged_probs: (N, C) array
    """
    stacked = np.stack(prob_list, axis=0)       # (K, N, C)
    averaged = np.mean(stacked, axis=0)         # (N, C)
    return averaged


def weighted_average(
    prob_list: List[np.ndarray],
    weights: List[float],
) -> np.ndarray:
    """
    Weighted Averaging: weighted sum of probability distributions.

    Weights are normalised to sum to 1.

    Args:
        prob_list: list of K arrays, each (N, C)
        weights:   list of K floats  (e.g. [0.4, 0.35, 0.25])

    Returns:
        weighted_probs: (N, C) array
    """
    weights = np.array(weights, dtype=np.float64)
    weights /= weights.sum()                     # normalise

    stacked = np.stack(prob_list, axis=0)         # (K, N, C)
    # Broadcast weights (K, 1, 1) over (K, N, C)
    weighted = stacked * weights[:, None, None]
    combined = weighted.sum(axis=0)              # (N, C)
    return combined


def rank_average(prob_list: List[np.ndarray]) -> np.ndarray:
    """
    Rank Averaging (bonus strategy): convert probabilities to ranks,
    average the ranks, then pick the class with the best average rank.

    This can be more robust when models have different calibration.

    Args:
        prob_list: list of K arrays, each (N, C)

    Returns:
        rank_averaged_probs: (N, C) — pseudo-probabilities from rank averaging
    """
    from scipy.stats import rankdata

    rank_list = []
    for probs in prob_list:
        ranks = np.apply_along_axis(rankdata, 1, probs)
        rank_list.append(ranks)

    stacked = np.stack(rank_list, axis=0)
    avg_ranks = np.mean(stacked, axis=0)

    # Normalise to [0, 1] per sample
    row_sums = avg_ranks.sum(axis=1, keepdims=True)
    normalised = avg_ranks / row_sums
    return normalised


# =====================================================================
#  Ensemble Runner
# =====================================================================

def load_cached_probabilities(prob_dir: str) -> Tuple[List[np.ndarray], List[str], np.ndarray]:
    """
    Load cached per-model probabilities from a directory.

    Returns:
        (prob_list, model_tags, ids)
    """
    ids = np.load(os.path.join(prob_dir, "sample_ids.npy"), allow_pickle=True)

    prob_list = []
    found_tags = []

    for tag in MODEL_TAGS:
        prob_path = os.path.join(prob_dir, f"{tag}_probs.npy")
        if os.path.exists(prob_path):
            probs = np.load(prob_path)
            prob_list.append(probs)
            found_tags.append(tag)
            logger.info(f"Loaded {tag} probs: shape={probs.shape}")

    if not prob_list:
        raise FileNotFoundError(
            f"No probability files (*_probs.npy) found in {prob_dir}. "
            "Run the 'predict' step first."
        )

    logger.info(f"Loaded {len(prob_list)} model predictions: {found_tags}")
    return prob_list, found_tags, ids


def run_ensemble(
    prob_dir: str,
    output_csv: str,
    strategy: str = "soft_vote",
    weights: Optional[List[float]] = None,
    task: str = "A",
) -> pd.DataFrame:
    """
    Run the ensemble strategy and write a submission CSV.

    Args:
        prob_dir:   directory containing cached *_probs.npy files
        output_csv: path to output submission CSV
        strategy:   one of 'soft_vote', 'weighted_avg', 'rank_avg'
        weights:    model weights for weighted_avg strategy
        task:       'A' or 'B' (for label mapping)

    Returns:
        DataFrame with ID, prediction columns
    """
    prob_list, found_tags, ids = load_cached_probabilities(prob_dir)

    # Validate shapes
    shapes = [p.shape for p in prob_list]
    if len(set(s[0] for s in shapes)) > 1:
        raise ValueError(f"Sample count mismatch across models: {shapes}")
    if len(set(s[1] for s in shapes)) > 1:
        raise ValueError(f"Class count mismatch across models: {shapes}")

    n_samples, n_classes = prob_list[0].shape
    logger.info(f"Ensemble: {len(prob_list)} models × {n_samples} samples × {n_classes} classes")

    # ── Apply strategy ──
    if strategy == "soft_vote":
        logger.info("Strategy: Soft Voting (uniform weights)")
        combined_probs = soft_vote(prob_list)

    elif strategy == "weighted_avg":
        if weights is None:
            # Default weights: equal
            weights = [1.0 / len(prob_list)] * len(prob_list)
        if len(weights) != len(prob_list):
            raise ValueError(
                f"Number of weights ({len(weights)}) must match "
                f"number of models ({len(prob_list)})"
            )
        weight_str = ", ".join(f"{t}={w:.3f}" for t, w in zip(found_tags, weights))
        logger.info(f"Strategy: Weighted Averaging ({weight_str})")
        combined_probs = weighted_average(prob_list, weights)

    elif strategy == "rank_avg":
        logger.info("Strategy: Rank Averaging")
        combined_probs = rank_average(prob_list)

    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use: soft_vote, weighted_avg, rank_avg")

    # ── Generate predictions ──
    predictions = combined_probs.argmax(axis=1)

    # ── Compute ensemble confidence stats ──
    max_probs = combined_probs.max(axis=1)
    logger.info(f"Ensemble confidence: mean={max_probs.mean():.4f}, "
                f"min={max_probs.min():.4f}, max={max_probs.max():.4f}")

    # ── Per-model agreement analysis ──
    per_model_preds = [p.argmax(axis=1) for p in prob_list]
    agreement = np.all(
        np.stack(per_model_preds, axis=0) == per_model_preds[0][None, :],
        axis=0
    )
    agreement_rate = agreement.mean()
    logger.info(f"Model agreement rate: {agreement_rate:.2%} "
                f"({agreement.sum()}/{n_samples} unanimous)")

    # ── Build submission DataFrame ──
    submission = pd.DataFrame({
        "ID": ids,
        "prediction": predictions.astype(int),
    })

    # Save
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    submission.to_csv(output_csv, index=False)
    logger.info(f"Submission saved → {output_csv}  ({len(submission)} rows)")

    # ── Save ensemble metadata ──
    meta = {
        "strategy": strategy,
        "models": found_tags,
        "weights": [float(w) for w in (weights or [1.0 / len(prob_list)] * len(prob_list))],
        "n_samples": int(n_samples),
        "n_classes": int(n_classes),
        "agreement_rate": float(agreement_rate),
        "mean_confidence": float(max_probs.mean()),
        "task": task,
    }
    meta_path = os.path.splitext(output_csv)[0] + "_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Metadata saved → {meta_path}")

    # ── Save combined probabilities for downstream analysis ──
    combined_path = os.path.join(
        os.path.dirname(output_csv) or ".",
        "ensemble_combined_probs.npy",
    )
    np.save(combined_path, combined_probs)
    logger.info(f"Combined probabilities saved → {combined_path}")

    return submission


# =====================================================================
#  Evaluation (when gold labels are available)
# =====================================================================

def evaluate_ensemble(
    submission_csv: str,
    gold_csv: str,
    task: str = "A",
) -> Dict[str, float]:
    """
    Evaluate ensemble predictions against gold labels.
    Computes Macro F1, Accuracy, Precision, and Recall.

    Args:
        submission_csv: path to ensemble submission CSV (ID, prediction)
        gold_csv:       path to gold labels CSV (ID, label)
        task:           'A' or 'B'

    Returns:
        dict of metrics
    """
    from sklearn.metrics import (
        f1_score, accuracy_score,
        precision_score, recall_score,
        classification_report,
    )

    pred_df = pd.read_csv(submission_csv)
    gold_df = pd.read_csv(gold_csv)

    merged = pd.merge(gold_df, pred_df, on="ID")
    if merged.empty:
        raise ValueError("No matching IDs between prediction and gold files.")

    y_true = merged["label"].values
    y_pred = merged["prediction"].values

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    logger.info("=" * 60)
    logger.info(f"  ENSEMBLE EVALUATION — Task {task}")
    logger.info("=" * 60)
    logger.info(f"  Macro F1:       {macro_f1:.4f}")
    logger.info(f"  Accuracy:       {accuracy:.4f}")
    logger.info(f"  Macro Precision:{macro_precision:.4f}")
    logger.info(f"  Macro Recall:   {macro_recall:.4f}")

    # Detailed report
    label_names = list(TASK_LABELS.get(task, {}).values())
    if label_names:
        report = classification_report(
            y_true, y_pred,
            target_names=label_names[:len(set(y_true))],
            zero_division=0,
        )
    else:
        report = classification_report(y_true, y_pred, zero_division=0)
    logger.info(f"\n{report}")
    logger.info("=" * 60)

    return {
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
    }


# =====================================================================
#  Weight Optimisation (optional: search for optimal weights on val set)
# =====================================================================

def optimize_weights(
    prob_dir: str,
    gold_csv: str,
    task: str = "A",
    n_steps: int = 21,
) -> Tuple[List[float], float]:
    """
    Grid search over 3-model weight combinations to maximise Macro F1.

    Args:
        prob_dir:  directory with cached *_probs.npy and sample_ids.npy
        gold_csv:  ground truth CSV with ID, label
        task:      'A' or 'B'
        n_steps:   number of steps per weight axis (granularity)

    Returns:
        (best_weights, best_macro_f1)
    """
    from sklearn.metrics import f1_score

    prob_list, found_tags, ids = load_cached_probabilities(prob_dir)

    if len(prob_list) < 2:
        logger.warning("Need at least 2 models for weight optimisation.")
        return [1.0], 0.0

    # Load gold labels
    gold_df = pd.read_csv(gold_csv)
    id_to_label = dict(zip(gold_df["ID"].astype(str), gold_df["label"]))

    # Align order
    y_true = np.array([id_to_label.get(str(id_), -1) for id_ in ids])
    valid_mask = y_true >= 0
    if valid_mask.sum() == 0:
        raise ValueError("No matching IDs between cached predictions and gold labels.")

    y_true = y_true[valid_mask]
    filtered_probs = [p[valid_mask] for p in prob_list]

    logger.info(f"Optimizing weights for {len(found_tags)} models on "
                f"{len(y_true)} samples (step={1.0/(n_steps-1):.3f}) …")

    best_f1 = -1.0
    best_weights = None
    n_models = len(filtered_probs)

    if n_models == 2:
        # 2-model grid search
        for w0 in np.linspace(0, 1, n_steps):
            w1 = 1.0 - w0
            combined = weighted_average(filtered_probs, [w0, w1])
            preds = combined.argmax(axis=1)
            f1 = f1_score(y_true, preds, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_weights = [w0, w1]

    elif n_models == 3:
        # 3-model simplex grid search
        for w0 in np.linspace(0, 1, n_steps):
            for w1 in np.linspace(0, 1 - w0, max(2, int(n_steps * (1 - w0)))):
                w2 = 1.0 - w0 - w1
                if w2 < 0:
                    continue
                combined = weighted_average(filtered_probs, [w0, w1, w2])
                preds = combined.argmax(axis=1)
                f1 = f1_score(y_true, preds, average="macro", zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_weights = [w0, w1, w2]
    else:
        # For >3 models, fall back to random search
        logger.info(f"Using random search for {n_models} models (1000 trials) …")
        rng = np.random.default_rng(42)
        for _ in range(1000):
            raw = rng.dirichlet(np.ones(n_models))
            combined = weighted_average(filtered_probs, raw.tolist())
            preds = combined.argmax(axis=1)
            f1 = f1_score(y_true, preds, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_weights = raw.tolist()

    weight_str = ", ".join(f"{t}={w:.3f}" for t, w in zip(found_tags, best_weights))
    logger.info(f"Best weights: {weight_str}")
    logger.info(f"Best Macro F1: {best_f1:.4f}")

    return best_weights, best_f1


# =====================================================================
#  Full Pipeline
# =====================================================================

def run_full_pipeline(
    model_paths: List[str],
    parquet_path: str,
    output_dir: str,
    output_csv: str,
    task: str = "A",
    strategy: str = "soft_vote",
    weights: Optional[List[float]] = None,
    max_length: int = 512,
    batch_size: int = 32,
    device: str = "cuda",
    gold_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    End-to-end ensemble pipeline:
      1. Run inference for each model
      2. Optionally optimise weights on validation set
      3. Combine predictions via chosen strategy
      4. Optionally evaluate against gold labels
    """
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available — falling back to CPU")
        device = "cpu"

    # Determine model tags
    tags = []
    for i, mp in enumerate(model_paths):
        tag = MODEL_TAGS[i] if i < len(MODEL_TAGS) else f"model_{i}"
        tags.append(tag)

    # ── Step 1: Per-model inference ──
    logger.info("=" * 60)
    logger.info("  STEP 1: Per-Model Inference")
    logger.info("=" * 60)

    all_probs = []
    all_ids = None

    for path, tag in zip(model_paths, tags):
        probs, ids = generate_model_predictions(
            model_path=path,
            parquet_path=parquet_path,
            output_dir=output_dir,
            model_tag=tag,
            max_length=max_length,
            batch_size=batch_size,
            device=device,
        )
        all_probs.append(probs)
        all_ids = ids

    # ── Step 2: Optimise weights (optional) ──
    if strategy == "weighted_avg" and weights is None and gold_csv is not None:
        logger.info("=" * 60)
        logger.info("  STEP 2: Weight Optimisation")
        logger.info("=" * 60)
        weights, _ = optimize_weights(output_dir, gold_csv, task=task)
    elif strategy == "weighted_avg" and weights is None:
        logger.info("No gold labels provided — using equal weights.")
        weights = [1.0 / len(all_probs)] * len(all_probs)

    # ── Step 3: Ensemble ──
    logger.info("=" * 60)
    logger.info("  STEP 3: Ensemble Combination")
    logger.info("=" * 60)

    submission = run_ensemble(
        prob_dir=output_dir,
        output_csv=output_csv,
        strategy=strategy,
        weights=weights,
        task=task,
    )

    # ── Step 4: Evaluate (optional) ──
    if gold_csv is not None:
        logger.info("=" * 60)
        logger.info("  STEP 4: Evaluation")
        logger.info("=" * 60)
        evaluate_ensemble(output_csv, gold_csv, task=task)

    return submission


# =====================================================================
#  CLI
# =====================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ensemble pipeline for CodeBERT + GraphCodeBERT + UniXcoder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-command to run")

    # ── predict ──
    pred = subparsers.add_parser("predict", help="Generate per-model softmax probabilities")
    pred.add_argument("--model_paths", nargs="+", required=True,
                       help="Paths to fine-tuned model directories")
    pred.add_argument("--model_tags", nargs="+", default=None,
                       help="Tags for each model (default: codebert, graphcodebert, unixcoder)")
    pred.add_argument("--parquet_path", required=True,
                       help="Path to test parquet file")
    pred.add_argument("--output_dir", required=True,
                       help="Directory to cache probability .npy files")
    pred.add_argument("--task", choices=["A", "B"], default="A")
    pred.add_argument("--max_length", type=int, default=512)
    pred.add_argument("--batch_size", type=int, default=32)
    pred.add_argument("--device", default=None)

    # ── ensemble ──
    ens = subparsers.add_parser("ensemble", help="Combine cached predictions")
    ens.add_argument("--prob_dir", required=True,
                      help="Directory containing *_probs.npy files")
    ens.add_argument("--output_csv", required=True,
                      help="Output submission CSV path")
    ens.add_argument("--strategy", choices=["soft_vote", "weighted_avg", "rank_avg"],
                      default="soft_vote")
    ens.add_argument("--weights", nargs="+", type=float, default=None,
                      help="Model weights for weighted_avg (must match #models)")
    ens.add_argument("--task", choices=["A", "B"], default="A")
    ens.add_argument("--gold_csv", default=None,
                      help="Gold labels CSV for evaluation (optional)")

    # ── optimize ──
    opt = subparsers.add_parser("optimize", help="Find optimal weights on validation set")
    opt.add_argument("--prob_dir", required=True,
                      help="Directory containing *_probs.npy files")
    opt.add_argument("--gold_csv", required=True,
                      help="Gold labels CSV (ID, label)")
    opt.add_argument("--task", choices=["A", "B"], default="A")
    opt.add_argument("--n_steps", type=int, default=21,
                      help="Grid search granularity")

    # ── full ──
    full = subparsers.add_parser("full", help="Full pipeline: predict + ensemble")
    full.add_argument("--model_paths", nargs="+", required=True)
    full.add_argument("--parquet_path", required=True)
    full.add_argument("--output_dir", required=True)
    full.add_argument("--output_csv", required=True)
    full.add_argument("--task", choices=["A", "B"], default="A")
    full.add_argument("--strategy", choices=["soft_vote", "weighted_avg", "rank_avg"],
                       default="soft_vote")
    full.add_argument("--weights", nargs="+", type=float, default=None)
    full.add_argument("--max_length", type=int, default=512)
    full.add_argument("--batch_size", type=int, default=32)
    full.add_argument("--device", default=None)
    full.add_argument("--gold_csv", default=None)

    # ── evaluate ──
    ev = subparsers.add_parser("evaluate", help="Evaluate submission against gold labels")
    ev.add_argument("--submission_csv", required=True)
    ev.add_argument("--gold_csv", required=True)
    ev.add_argument("--task", choices=["A", "B"], default="A")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "predict":
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        tags = args.model_tags or MODEL_TAGS[: len(args.model_paths)]

        for path, tag in zip(args.model_paths, tags):
            generate_model_predictions(
                model_path=path,
                parquet_path=args.parquet_path,
                output_dir=args.output_dir,
                model_tag=tag,
                max_length=args.max_length,
                batch_size=args.batch_size,
                device=device,
            )

    elif args.command == "ensemble":
        submission = run_ensemble(
            prob_dir=args.prob_dir,
            output_csv=args.output_csv,
            strategy=args.strategy,
            weights=args.weights,
            task=args.task,
        )
        if args.gold_csv:
            evaluate_ensemble(args.output_csv, args.gold_csv, task=args.task)

    elif args.command == "optimize":
        best_w, best_f1 = optimize_weights(
            prob_dir=args.prob_dir,
            gold_csv=args.gold_csv,
            task=args.task,
            n_steps=args.n_steps,
        )
        print(f"\nOptimal weights: {best_w}")
        print(f"Optimal Macro F1: {best_f1:.4f}")

    elif args.command == "full":
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        run_full_pipeline(
            model_paths=args.model_paths,
            parquet_path=args.parquet_path,
            output_dir=args.output_dir,
            output_csv=args.output_csv,
            task=args.task,
            strategy=args.strategy,
            weights=args.weights,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=device,
            gold_csv=args.gold_csv,
        )

    elif args.command == "evaluate":
        evaluate_ensemble(
            submission_csv=args.submission_csv,
            gold_csv=args.gold_csv,
            task=args.task,
        )


if __name__ == "__main__":
    main()
