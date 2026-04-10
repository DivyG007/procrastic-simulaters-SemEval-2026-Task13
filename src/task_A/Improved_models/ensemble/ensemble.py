"""
ensemble.py — Unified Ensemble Pipeline for Task A
====================================================
Single-file pipeline that, with one call on Kaggle, does everything:

  1. Trains all 3 models sequentially (CodeBERT, GraphCodeBERT, UniXcoder)
  2. Loads cached softmax probability .npy files from each model
  3. Runs ensemble combination (soft voting / weighted average / rank average)
  4. Evaluates ensemble predictions (4-category breakdown if labels available)
  5. Saves submission CSV + metadata JSON

Usage on Kaggle:
    python ensemble.py                          # soft vote (default)
    python ensemble.py --strategy weighted_avg  # weighted average
    python ensemble.py --strategy weighted_avg --weights 0.4 0.35 0.25
    python ensemble.py --skip_training          # skip training, use cached probs

Ensemble Strategies:
  soft_vote     — average class probabilities, then argmax
  weighted_avg  — weighted sum of class probabilities, then argmax
  rank_avg      — rank averaging (robust to different calibration)
"""

import os
import sys
import json
import argparse
import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Ensure local imports work when run as a script ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MODEL_REGISTRY,
    ENSEMBLE_CACHE_DIR,
    ENSEMBLE_CSV,
    OUTPUT_ROOT,
    LABEL_MAP,
    NUM_LABELS,
    TEST_PARQUET,
)

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ensemble")

# ── Model tags (order matters for default weights) ─────────────────────
MODEL_TAGS = list(MODEL_REGISTRY.keys())  # ["codebert", "graphcodebert", "unixcoder"]


# =====================================================================
#  Step 1 — Train All Models
# =====================================================================

def train_all_models():
    """
    Train CodeBERT, GraphCodeBERT, and UniXcoder sequentially.

    Each training script saves:
      - Model checkpoint in /kaggle/working/{tag}_deep_head/
      - Probability .npy in /kaggle/working/ensemble_cache/{tag}_probs.npy
    """
    # Import training scripts (they expose a main() function)
    import train_codebert
    import train_graphcodebert
    import train_unixcoder

    trainers = [
        ("codebert",      train_codebert),
        ("graphcodebert", train_graphcodebert),
        ("unixcoder",     train_unixcoder),
    ]

    for tag, module in trainers:
        logger.info("=" * 70)
        logger.info(f"  TRAINING MODEL: {tag}")
        logger.info("=" * 70)
        module.main()
        logger.info(f"  ✓ {tag} training complete.\n")


# =====================================================================
#  Step 2 — Load Cached Probabilities
# =====================================================================

def load_cached_probabilities(
    prob_dir: str = ENSEMBLE_CACHE_DIR,
) -> Tuple[List[np.ndarray], List[str], np.ndarray]:
    """
    Load per-model softmax probabilities from the cache directory.

    Returns:
        (prob_list, found_tags, sample_ids)
    """
    ids = np.load(os.path.join(prob_dir, "sample_ids.npy"), allow_pickle=True)

    prob_list  = []
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
            "Run training first or check the cache directory."
        )

    logger.info(f"Loaded {len(prob_list)} model predictions: {found_tags}")
    return prob_list, found_tags, ids


# =====================================================================
#  Step 3 — Ensemble Strategies
# =====================================================================

def soft_vote(prob_list: List[np.ndarray]) -> np.ndarray:
    """
    Soft Voting: simple average of probability distributions.

    Each model gets equal weight.  Final prediction = argmax of mean probs.

    Args:
        prob_list: list of K arrays, each (N, C) softmax probabilities

    Returns:
        averaged_probs: (N, C) array
    """
    stacked  = np.stack(prob_list, axis=0)   # (K, N, C)
    averaged = np.mean(stacked, axis=0)      # (N, C)
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
        weights:   list of K floats (e.g. [0.4, 0.35, 0.25])

    Returns:
        weighted_probs: (N, C) array
    """
    weights = np.array(weights, dtype=np.float64)
    weights /= weights.sum()                   # normalise

    stacked  = np.stack(prob_list, axis=0)     # (K, N, C)
    weighted = stacked * weights[:, None, None]
    combined = weighted.sum(axis=0)            # (N, C)
    return combined


def rank_average(prob_list: List[np.ndarray]) -> np.ndarray:
    """
    Rank Averaging: convert probabilities to ranks, average the ranks,
    then pick the class with the best average rank.

    More robust when models have different calibration.

    Args:
        prob_list: list of K arrays, each (N, C)

    Returns:
        rank_averaged_probs: (N, C) pseudo-probabilities
    """
    from scipy.stats import rankdata

    rank_list = []
    for probs in prob_list:
        ranks = np.apply_along_axis(rankdata, 1, probs)
        rank_list.append(ranks)

    stacked   = np.stack(rank_list, axis=0)
    avg_ranks = np.mean(stacked, axis=0)

    # Normalise to [0, 1] per sample
    row_sums   = avg_ranks.sum(axis=1, keepdims=True)
    normalised = avg_ranks / row_sums
    return normalised


# =====================================================================
#  Step 4 — Run Ensemble
# =====================================================================

def run_ensemble(
    prob_dir: str = ENSEMBLE_CACHE_DIR,
    output_csv: str = ENSEMBLE_CSV,
    strategy: str = "soft_vote",
    weights: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Combine cached per-model probabilities and write a submission CSV.

    Args:
        prob_dir:   directory containing cached *_probs.npy files
        output_csv: path to output submission CSV
        strategy:   'soft_vote', 'weighted_avg', or 'rank_avg'
        weights:    model weights for weighted_avg (must match # models)

    Returns:
        DataFrame with ID, prediction columns
    """
    prob_list, found_tags, ids = load_cached_probabilities(prob_dir)

    # ── Validate shapes ──
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

    # ── Ensemble confidence stats ──
    max_probs = combined_probs.max(axis=1)
    logger.info(
        f"Ensemble confidence: mean={max_probs.mean():.4f}, "
        f"min={max_probs.min():.4f}, max={max_probs.max():.4f}"
    )

    # ── Per-model agreement analysis ──
    per_model_preds = [p.argmax(axis=1) for p in prob_list]
    agreement = np.all(
        np.stack(per_model_preds, axis=0) == per_model_preds[0][None, :],
        axis=0,
    )
    agreement_rate = agreement.mean()
    logger.info(
        f"Model agreement rate: {agreement_rate:.2%} "
        f"({agreement.sum()}/{n_samples} unanimous)"
    )

    # ── Build submission DataFrame ──
    submission = pd.DataFrame({
        "ID": ids,
        "prediction": predictions.astype(int),
    })

    # ── Save submission CSV ──
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
        "label_map": LABEL_MAP,
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
#  Step 5 — Evaluate Ensemble (if gold labels available)
# =====================================================================

def evaluate_ensemble_predictions(
    submission_csv: str,
    gold_parquet: str = TEST_PARQUET,
):
    """
    Evaluate ensemble predictions against gold labels from the test parquet.

    Prints Macro F1, Accuracy, per-class report, and 4-category breakdown.
    """
    from sklearn.metrics import (
        f1_score,
        accuracy_score,
        precision_score,
        recall_score,
        classification_report,
    )
    from train_utils import evaluate_by_category

    if not os.path.exists(gold_parquet):
        logger.warning(f"Gold parquet not found at {gold_parquet}, skipping evaluation.")
        return

    # Load predictions and gold
    pred_df = pd.read_csv(submission_csv)
    gold_df = pd.read_parquet(gold_parquet)
    gold_df = gold_df.dropna(subset=["code"]).reset_index(drop=True)

    if "label" not in gold_df.columns:
        logger.warning("Gold parquet has no 'label' column, skipping evaluation.")
        return

    gold_df["label"] = gold_df["label"].astype(int)

    # Determine ID column for merging
    if "ID" in gold_df.columns:
        gold_df["ID"] = gold_df["ID"].astype(str)
        pred_df["ID"] = pred_df["ID"].astype(str)
        merged = pd.merge(gold_df, pred_df, on="ID")
    else:
        # Fall back to positional alignment
        if len(gold_df) != len(pred_df):
            logger.warning(
                f"Cannot align: gold has {len(gold_df)} rows, "
                f"predictions have {len(pred_df)} rows."
            )
            return
        merged = gold_df.copy()
        merged["prediction"] = pred_df["prediction"].values

    if merged.empty:
        logger.warning("No matching samples found between predictions and gold labels.")
        return

    y_true = merged["label"].values
    y_pred = merged["prediction"].values

    # Compute metrics
    macro_f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    accuracy  = accuracy_score(y_true, y_pred)
    macro_p   = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r   = recall_score(y_true, y_pred, average="macro", zero_division=0)

    logger.info("=" * 70)
    logger.info("  ENSEMBLE EVALUATION — Task A")
    logger.info("=" * 70)
    logger.info(f"  Macro F1:        {macro_f1:.4f}")
    logger.info(f"  Accuracy:        {accuracy:.4f}")
    logger.info(f"  Macro Precision: {macro_p:.4f}")
    logger.info(f"  Macro Recall:    {macro_r:.4f}")

    label_names = list(LABEL_MAP.values())
    report = classification_report(
        y_true, y_pred,
        target_names=label_names[:len(set(y_true))],
        zero_division=0,
    )
    logger.info(f"\n{report}")
    logger.info("=" * 70)

    # 4-category breakdown
    evaluate_by_category(merged, tag="Ensemble")


# =====================================================================
#  Weight Optimisation (grid search over 3-model weights)
# =====================================================================

def optimize_weights(
    prob_dir: str = ENSEMBLE_CACHE_DIR,
    gold_labels: Optional[np.ndarray] = None,
    n_steps: int = 21,
) -> Tuple[List[float], float]:
    """
    Grid search over 3-model weight combinations to maximise Macro F1.

    Args:
        prob_dir:     directory with cached *_probs.npy and sample_ids.npy
        gold_labels:  array of ground-truth labels aligned with sample_ids
        n_steps:      number of steps per weight axis (granularity)

    Returns:
        (best_weights, best_macro_f1)
    """
    from sklearn.metrics import f1_score

    prob_list, found_tags, ids = load_cached_probabilities(prob_dir)

    if gold_labels is None:
        logger.warning("No gold labels provided for weight optimisation.")
        return [1.0 / len(prob_list)] * len(prob_list), 0.0

    if len(prob_list) < 2:
        logger.warning("Need at least 2 models for weight optimisation.")
        return [1.0], 0.0

    y_true = gold_labels
    logger.info(
        f"Optimizing weights for {len(found_tags)} models on "
        f"{len(y_true)} samples (step={1.0/(n_steps-1):.3f}) …"
    )

    best_f1      = -1.0
    best_weights = None
    n_models     = len(prob_list)

    if n_models == 2:
        for w0 in np.linspace(0, 1, n_steps):
            w1 = 1.0 - w0
            combined = weighted_average(prob_list, [w0, w1])
            preds    = combined.argmax(axis=1)
            f1 = f1_score(y_true, preds, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1      = f1
                best_weights = [w0, w1]

    elif n_models == 3:
        for w0 in np.linspace(0, 1, n_steps):
            for w1 in np.linspace(0, 1 - w0, max(2, int(n_steps * (1 - w0)))):
                w2 = 1.0 - w0 - w1
                if w2 < 0:
                    continue
                combined = weighted_average(prob_list, [w0, w1, w2])
                preds    = combined.argmax(axis=1)
                f1 = f1_score(y_true, preds, average="macro", zero_division=0)
                if f1 > best_f1:
                    best_f1      = f1
                    best_weights = [w0, w1, w2]
    else:
        # For >3 models, random search with Dirichlet sampling
        logger.info(f"Using random search for {n_models} models (1000 trials) …")
        rng = np.random.default_rng(42)
        for _ in range(1000):
            raw      = rng.dirichlet(np.ones(n_models))
            combined = weighted_average(prob_list, raw.tolist())
            preds    = combined.argmax(axis=1)
            f1 = f1_score(y_true, preds, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1      = f1
                best_weights = raw.tolist()

    weight_str = ", ".join(f"{t}={w:.3f}" for t, w in zip(found_tags, best_weights))
    logger.info(f"Best weights: {weight_str}")
    logger.info(f"Best Macro F1: {best_f1:.4f}")

    return best_weights, best_f1


# =====================================================================
#  CLI  &  Main Entry Point
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Task A Ensemble Pipeline — trains CodeBERT + GraphCodeBERT + UniXcoder, "
            "then combines predictions via soft voting or weighted averaging."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--strategy",
        choices=["soft_vote", "weighted_avg", "rank_avg"],
        default="soft_vote",
        help="Ensemble strategy (default: soft_vote)",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Model weights for weighted_avg (e.g. --weights 0.4 0.35 0.25)",
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training; use already-cached probability .npy files",
    )
    parser.add_argument(
        "--output_csv",
        default=ENSEMBLE_CSV,
        help=f"Path to save the ensemble submission CSV (default: {ENSEMBLE_CSV})",
    )
    parser.add_argument(
        "--cache_dir",
        default=ENSEMBLE_CACHE_DIR,
        help=f"Directory for cached probability .npy files (default: {ENSEMBLE_CACHE_DIR})",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 70)
    logger.info("  TASK A — ENSEMBLE PIPELINE")
    logger.info("=" * 70)
    logger.info(f"  Strategy:       {args.strategy}")
    logger.info(f"  Skip training:  {args.skip_training}")
    logger.info(f"  Cache dir:      {args.cache_dir}")
    logger.info(f"  Output CSV:     {args.output_csv}")
    if args.weights:
        logger.info(f"  Weights:        {args.weights}")
    logger.info("=" * 70)

    # ── Step 1: Train all 3 models (unless --skip_training) ──
    if not args.skip_training:
        logger.info("\n" + "=" * 70)
        logger.info("  STEP 1: SEQUENTIAL MODEL TRAINING")
        logger.info("=" * 70 + "\n")
        train_all_models()
    else:
        logger.info("Skipping training — using cached probabilities.")

    # ── Step 2: Ensemble combination ──
    logger.info("\n" + "=" * 70)
    logger.info("  STEP 2: ENSEMBLE COMBINATION")
    logger.info("=" * 70 + "\n")

    submission = run_ensemble(
        prob_dir=args.cache_dir,
        output_csv=args.output_csv,
        strategy=args.strategy,
        weights=args.weights,
    )

    # ── Step 3: Evaluate (if gold labels available) ──
    logger.info("\n" + "=" * 70)
    logger.info("  STEP 3: ENSEMBLE EVALUATION")
    logger.info("=" * 70 + "\n")

    evaluate_ensemble_predictions(
        submission_csv=args.output_csv,
        gold_parquet=TEST_PARQUET,
    )

    # ── Done ──
    logger.info("\n" + "=" * 70)
    logger.info("  PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Submission:  {args.output_csv}")
    logger.info(f"  Metadata:    {os.path.splitext(args.output_csv)[0]}_meta.json")
    logger.info(f"  Models used: {list(MODEL_REGISTRY.keys())}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
