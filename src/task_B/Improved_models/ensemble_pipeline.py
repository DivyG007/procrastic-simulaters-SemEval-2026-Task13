"""Task B Ensemble Pipeline (CLI)

Combines probabilities from multiple fine-tuned models using:
- `soft_vote`
- `weighted_avg`
- `rank_avg`

Subcommands:
- `predict`   : cache per-model probabilities
- `ensemble`  : combine cached probabilities into submission
- `optimize`  : search best weights on labeled set
- `full`      : predict + ensemble (+ optional optimize/evaluate)
- `evaluate`  : evaluate a submission CSV vs gold labels

Example:
    python3 ensemble_pipeline.py full \
      --model_paths /path/codebert /path/graphcodebert /path/unixcoder \
      --parquet_path /path/task_b_validation_set.parquet \
      --gold_csv /path/task_b_validation_labels.csv \
      --output_dir ./ensemble_cache_taskB \
      --output_csv ./submission_b_ensemble.csv \
      --strategy weighted_avg \
      --task B
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Suppress TF/XLA noise in notebook/colab/kaggle environments.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["USE_TF"] = "0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("taskb_ensemble")

TASK_LABELS = {
    "A": {0: "human", 1: "machine"},
    "B": {
        0: "human",
        1: "deepseek",
        2: "qwen",
        3: "01-ai",
        4: "bigcode",
        5: "gemma",
        6: "phi",
        7: "meta-llama",
        8: "ibm-granite",
        9: "mistral",
        10: "openai",
    },
}


def infer_model_tag(model_path: str, index: int) -> str:
    """Infer a readable model tag from model path."""
    name = os.path.basename(model_path.rstrip("/")).lower()
    if "graphcodebert" in name:
        return "graphcodebert"
    if "unixcoder" in name:
        return "unixcoder"
    if "codebert" in name:
        return "codebert"
    return f"model_{index}"


def load_model_and_tokenizer(model_path: str, device: str):
    """Load tokenizer/model from a HuggingFace-compatible checkpoint."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    logger.info(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    logger.info(f"  -> num_labels={model.config.num_labels} | device={device}")
    return model, tokenizer


@torch.no_grad()
def predict_probabilities(
    model,
    tokenizer,
    codes: List[str],
    max_length: int,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """Return softmax probability matrix of shape `(N, C)` for `codes`."""
    chunks: List[np.ndarray] = []
    for i in tqdm(range(0, len(codes), batch_size), desc="  Inference"):
        batch = codes[i : i + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        logits = model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
        ).logits
        chunks.append(F.softmax(logits, dim=-1).cpu().numpy())

    return np.concatenate(chunks, axis=0)


def _load_parquet_for_inference(parquet_path: str) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Load parquet and return `(df, ids, codes)` with cleaned rows."""
    df = pd.read_parquet(parquet_path)
    if "code" not in df.columns:
        raise ValueError(f"'{parquet_path}' must contain a 'code' column")

    df = df.dropna(subset=["code"]).reset_index(drop=True)
    codes = df["code"].astype(str).tolist()

    if "ID" in df.columns:
        ids = df["ID"].astype(str).tolist()
    elif "id" in df.columns:
        ids = df["id"].astype(str).tolist()
    else:
        ids = [str(i) for i in range(len(df))]

    return df, ids, codes


def generate_model_predictions(
    model_path: str,
    parquet_path: str,
    output_dir: str,
    model_tag: str,
    max_length: int,
    batch_size: int,
    device: str,
) -> Tuple[np.ndarray, List[str]]:
    """Run single-model inference and cache output probabilities as `.npy`."""
    os.makedirs(output_dir, exist_ok=True)

    _, ids, codes = _load_parquet_for_inference(parquet_path)
    model, tokenizer = load_model_and_tokenizer(model_path, device)

    logger.info(f"[{model_tag}] Predicting on {len(codes)} samples")
    probs = predict_probabilities(model, tokenizer, codes, max_length, batch_size, device)

    prob_path = os.path.join(output_dir, f"{model_tag}_probs.npy")
    id_path = os.path.join(output_dir, "sample_ids.npy")
    np.save(prob_path, probs)
    np.save(id_path, np.array(ids))

    logger.info(f"[{model_tag}] Saved -> {prob_path} shape={probs.shape}")

    del model, tokenizer
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return probs, ids


def soft_vote(prob_list: List[np.ndarray]) -> np.ndarray:
    """Simple mean probability fusion."""
    return np.mean(np.stack(prob_list, axis=0), axis=0)


def weighted_average(prob_list: List[np.ndarray], weights: List[float]) -> np.ndarray:
    """Weighted probability fusion with normalized weights."""
    w = np.array(weights, dtype=np.float64)
    w = w / w.sum()
    arr = np.stack(prob_list, axis=0)
    return (arr * w[:, None, None]).sum(axis=0)


def rank_average(prob_list: List[np.ndarray]) -> np.ndarray:
    """Rank-based fusion, robust to calibration mismatch."""
    from scipy.stats import rankdata

    ranks = [np.apply_along_axis(rankdata, 1, p) for p in prob_list]
    avg = np.mean(np.stack(ranks, axis=0), axis=0)
    return avg / avg.sum(axis=1, keepdims=True)


def load_cached_probabilities(prob_dir: str) -> Tuple[List[np.ndarray], List[str], np.ndarray]:
    """Load all `*_probs.npy` and shared `sample_ids.npy` from cache folder."""
    id_path = os.path.join(prob_dir, "sample_ids.npy")
    if not os.path.exists(id_path):
        raise FileNotFoundError(f"Missing {id_path}")

    ids = np.load(id_path, allow_pickle=True)

    prob_files = [f for f in os.listdir(prob_dir) if f.endswith("_probs.npy")]
    prob_files = sorted(prob_files)
    if not prob_files:
        raise FileNotFoundError(f"No *_probs.npy files found in {prob_dir}")

    prob_list: List[np.ndarray] = []
    tags: List[str] = []

    for filename in prob_files:
        tag = filename.replace("_probs.npy", "")
        path = os.path.join(prob_dir, filename)
        probs = np.load(path)
        prob_list.append(probs)
        tags.append(tag)
        logger.info(f"Loaded {tag}: shape={probs.shape}")

    shapes = [p.shape for p in prob_list]
    if len({s[0] for s in shapes}) != 1:
        raise ValueError(f"Sample-count mismatch across models: {shapes}")
    if len({s[1] for s in shapes}) != 1:
        raise ValueError(f"Class-count mismatch across models: {shapes}")

    return prob_list, tags, ids


def run_ensemble(
    prob_dir: str,
    output_csv: str,
    strategy: str,
    weights: Optional[List[float]],
    task: str,
) -> pd.DataFrame:
    """Run chosen fusion strategy over cached probabilities and save CSV."""
    prob_list, tags, ids = load_cached_probabilities(prob_dir)

    if strategy == "soft_vote":
        combined = soft_vote(prob_list)
        used_weights = [1.0 / len(prob_list)] * len(prob_list)

    elif strategy == "weighted_avg":
        if weights is None:
            weights = [1.0 / len(prob_list)] * len(prob_list)
        if len(weights) != len(prob_list):
            raise ValueError("Number of weights must match number of model probability files")
        combined = weighted_average(prob_list, weights)
        used_weights = [float(x) for x in weights]

    elif strategy == "rank_avg":
        combined = rank_average(prob_list)
        used_weights = [1.0 / len(prob_list)] * len(prob_list)

    else:
        raise ValueError("Unknown strategy. Use: soft_vote | weighted_avg | rank_avg")

    preds = combined.argmax(axis=1).astype(int)
    confidence = combined.max(axis=1)

    per_model_preds = [p.argmax(axis=1) for p in prob_list]
    agreement = np.all(np.stack(per_model_preds, axis=0) == per_model_preds[0][None, :], axis=0)

    out = pd.DataFrame({"ID": ids, "prediction": preds})
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    out.to_csv(output_csv, index=False)
    logger.info(f"Saved submission -> {output_csv} ({len(out)} rows)")

    meta = {
        "task": task,
        "strategy": strategy,
        "models": tags,
        "weights": used_weights,
        "n_samples": int(combined.shape[0]),
        "n_classes": int(combined.shape[1]),
        "mean_confidence": float(confidence.mean()),
        "agreement_rate": float(agreement.mean()),
    }
    meta_path = os.path.splitext(output_csv)[0] + "_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved metadata -> {meta_path}")

    np.save(os.path.join(os.path.dirname(output_csv) or ".", "ensemble_combined_probs.npy"), combined)

    return out


def evaluate_ensemble(submission_csv: str, gold_csv: str, task: str) -> Dict[str, float]:
    """Evaluate submission file against gold labels and print report."""
    from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

    pred_df = pd.read_csv(submission_csv)
    gold_df = pd.read_csv(gold_csv)
    merged = pd.merge(gold_df, pred_df, on="ID", how="inner")

    if merged.empty:
        raise ValueError("No overlapping IDs between prediction and gold files")

    y_true = merged["label"].to_numpy()
    y_pred = merged["prediction"].to_numpy()

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    logger.info(f"Macro F1={macro_f1:.4f} | Accuracy={acc:.4f} | Macro P={macro_precision:.4f} | Macro R={macro_recall:.4f}")

    labels = TASK_LABELS.get(task, {})
    if labels:
        print(classification_report(y_true, y_pred, labels=sorted(labels.keys()), target_names=[labels[i] for i in sorted(labels.keys())], zero_division=0))
    else:
        print(classification_report(y_true, y_pred, zero_division=0))

    return {
        "macro_f1": macro_f1,
        "accuracy": acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
    }


def optimize_weights(prob_dir: str, gold_csv: str, n_steps: int = 21) -> Tuple[List[float], float]:
    """Search ensemble weights maximizing macro-F1 on a labeled set."""
    from sklearn.metrics import f1_score

    prob_list, tags, ids = load_cached_probabilities(prob_dir)
    gold_df = pd.read_csv(gold_csv)
    id_to_label = dict(zip(gold_df["ID"].astype(str), gold_df["label"]))
    y_true = np.array([id_to_label.get(str(i), -1) for i in ids])

    mask = y_true >= 0
    if mask.sum() == 0:
        raise ValueError("No overlapping IDs for optimization")

    y_true = y_true[mask]
    probs = [p[mask] for p in prob_list]

    n_models = len(probs)
    best_w: Optional[List[float]] = None
    best_f1 = -1.0

    if n_models == 2:
        for w0 in np.linspace(0, 1, n_steps):
            w = [float(w0), float(1 - w0)]
            pred = weighted_average(probs, w).argmax(axis=1)
            f1 = f1_score(y_true, pred, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_w = w

    elif n_models == 3:
        for w0 in np.linspace(0, 1, n_steps):
            for w1 in np.linspace(0, 1 - w0, max(2, int(n_steps * (1 - w0)))):
                w2 = 1 - w0 - w1
                if w2 < 0:
                    continue
                w = [float(w0), float(w1), float(w2)]
                pred = weighted_average(probs, w).argmax(axis=1)
                f1 = f1_score(y_true, pred, average="macro", zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_w = w

    else:
        rng = np.random.default_rng(42)
        for _ in range(1000):
            w = rng.dirichlet(np.ones(n_models)).tolist()
            pred = weighted_average(probs, w).argmax(axis=1)
            f1 = f1_score(y_true, pred, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_w = w

    assert best_w is not None
    logger.info("Best weights: " + ", ".join(f"{t}={w:.3f}" for t, w in zip(tags, best_w)))
    logger.info(f"Best macro-F1: {best_f1:.4f}")
    return best_w, best_f1


def run_full_pipeline(
    model_paths: List[str],
    parquet_path: str,
    output_dir: str,
    output_csv: str,
    task: str,
    strategy: str,
    weights: Optional[List[float]],
    max_length: int,
    batch_size: int,
    device: str,
    gold_csv: Optional[str],
    model_tags: Optional[List[str]],
) -> pd.DataFrame:
    """Run full workflow: per-model prediction -> ensemble -> optional eval."""
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA unavailable, using CPU")
        device = "cpu"

    tags = model_tags or [infer_model_tag(p, i) for i, p in enumerate(model_paths)]
    if len(tags) != len(model_paths):
        raise ValueError("--model_tags count must match --model_paths count")

    logger.info("STEP 1/3: per-model inference")
    for i, (model_path, tag) in enumerate(zip(model_paths, tags)):
        generate_model_predictions(
            model_path=model_path,
            parquet_path=parquet_path,
            output_dir=output_dir,
            model_tag=tag,
            max_length=max_length,
            batch_size=batch_size,
            device=device,
        )

    if strategy == "weighted_avg" and weights is None and gold_csv is not None:
        logger.info("STEP 2/3: weight optimization")
        weights, _ = optimize_weights(output_dir, gold_csv)

    logger.info("STEP 3/3: ensembling")
    submission = run_ensemble(
        prob_dir=output_dir,
        output_csv=output_csv,
        strategy=strategy,
        weights=weights,
        task=task,
    )

    if gold_csv:
        evaluate_ensemble(output_csv, gold_csv, task)

    return submission


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for all ensemble subcommands."""
    parser = argparse.ArgumentParser(description="Task B ensemble pipeline")
    sub = parser.add_subparsers(dest="command")

    # predict
    pred = sub.add_parser("predict", help="Generate per-model probability cache")
    pred.add_argument("--model_paths", nargs="+", required=True)
    pred.add_argument("--model_tags", nargs="+", default=None)
    pred.add_argument("--parquet_path", required=True)
    pred.add_argument("--output_dir", default="./ensemble_cache_taskB")
    pred.add_argument("--task", choices=["A", "B"], default="B")
    pred.add_argument("--max_length", type=int, default=512)
    pred.add_argument("--batch_size", type=int, default=32)
    pred.add_argument("--device", default=None)

    # ensemble
    ens = sub.add_parser("ensemble", help="Run ensemble from cached probabilities")
    ens.add_argument("--prob_dir", default="./ensemble_cache_taskB")
    ens.add_argument("--output_csv", default="./submission_b_ensemble.csv")
    ens.add_argument("--strategy", choices=["soft_vote", "weighted_avg", "rank_avg"], default="soft_vote")
    ens.add_argument("--weights", nargs="+", type=float, default=None)
    ens.add_argument("--task", choices=["A", "B"], default="B")
    ens.add_argument("--gold_csv", default=None)

    # optimize
    opt = sub.add_parser("optimize", help="Find best weights on labeled set")
    opt.add_argument("--prob_dir", default="./ensemble_cache_taskB")
    opt.add_argument("--gold_csv", required=True)
    opt.add_argument("--n_steps", type=int, default=21)

    # full
    full = sub.add_parser("full", help="Run predict + ensemble (+optional optimize/eval)")
    full.add_argument("--model_paths", nargs="+", required=True)
    full.add_argument("--model_tags", nargs="+", default=None)
    full.add_argument("--parquet_path", required=True)
    full.add_argument("--output_dir", default="./ensemble_cache_taskB")
    full.add_argument("--output_csv", default="./submission_b_ensemble.csv")
    full.add_argument("--task", choices=["A", "B"], default="B")
    full.add_argument("--strategy", choices=["soft_vote", "weighted_avg", "rank_avg"], default="soft_vote")
    full.add_argument("--weights", nargs="+", type=float, default=None)
    full.add_argument("--max_length", type=int, default=512)
    full.add_argument("--batch_size", type=int, default=32)
    full.add_argument("--device", default=None)
    full.add_argument("--gold_csv", default=None)

    # evaluate
    ev = sub.add_parser("evaluate", help="Evaluate submission against labels")
    ev.add_argument("--submission_csv", required=True)
    ev.add_argument("--gold_csv", required=True)
    ev.add_argument("--task", choices=["A", "B"], default="B")

    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    device = getattr(args, "device", None) or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.command == "predict":
        tags = args.model_tags or [infer_model_tag(p, i) for i, p in enumerate(args.model_paths)]
        if len(tags) != len(args.model_paths):
            raise ValueError("--model_tags count must match --model_paths count")

        for model_path, tag in zip(args.model_paths, tags):
            generate_model_predictions(
                model_path=model_path,
                parquet_path=args.parquet_path,
                output_dir=args.output_dir,
                model_tag=tag,
                max_length=args.max_length,
                batch_size=args.batch_size,
                device=device,
            )

    elif args.command == "ensemble":
        run_ensemble(
            prob_dir=args.prob_dir,
            output_csv=args.output_csv,
            strategy=args.strategy,
            weights=args.weights,
            task=args.task,
        )
        if args.gold_csv:
            evaluate_ensemble(args.output_csv, args.gold_csv, args.task)

    elif args.command == "optimize":
        best_w, best_f1 = optimize_weights(args.prob_dir, args.gold_csv, args.n_steps)
        print(f"Optimal weights: {best_w}")
        print(f"Optimal Macro F1: {best_f1:.4f}")

    elif args.command == "full":
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
            model_tags=args.model_tags,
        )

    elif args.command == "evaluate":
        evaluate_ensemble(args.submission_csv, args.gold_csv, args.task)


if __name__ == "__main__":
    main()
