"""
Ensemble Notebook Utilities
============================
Drop-in integration for Kaggle / Colab notebooks.

Import this module in your notebook and use the high-level API:

    from ensemble_notebook import EnsemblePredictor
    
    ensemble = EnsemblePredictor(task="A", device="cuda")
    
    # Add trained models
    ensemble.add_model("codebert",      trainer_obj_1)
    ensemble.add_model("graphcodebert", trainer_obj_2)
    ensemble.add_model("unixcoder",     trainer_obj_3)
    
    # Or add from saved model paths
    ensemble.add_model_from_path("codebert", "/path/to/codebert_weights")
    
    # Run ensemble on test data
    results = ensemble.predict(
        parquet_path="test.parquet",
        strategy="weighted_avg",
        weights=[0.4, 0.35, 0.25],
    )
    
    # Save submission
    ensemble.save_submission(results, "submission.csv")
    
    # Evaluate (if gold labels available)
    ensemble.evaluate(results, gold_path="gold.csv")
"""

import gc
import os

# Suppress XLA / TensorFlow and CUDA plugin registration warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
os.environ["USE_TF"] = "0"

import json
import logging
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger("ensemble_notebook")
logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(message)s", datefmt="%H:%M:%S")

# ── Task label maps ─────────────────────────────────────────────────────
TASK_LABELS = {
    "A": {0: "human", 1: "machine"},
    "B": {
        0: "human", 1: "deepseek", 2: "qwen", 3: "01-ai",
        4: "bigcode", 5: "gemma", 6: "phi", 7: "meta-llama",
        8: "ibm-granite", 9: "mistral", 10: "openai",
    },
}

# ── Evaluation settings ────────────────────────────────────────────────
SEEN_LANGS    = {"c++", "cpp", "python", "java"}
UNSEEN_LANGS  = {"go", "php", "c#", "csharp", "c", "javascript", "js"}
SEEN_DOMAINS  = {"algorithmic"}
UNSEEN_DOMAINS = {"research", "production"}


class EnsemblePredictor:
    """
    High-level ensemble predictor for CodeBERT / GraphCodeBERT / UniXcoder.
    
    Supports:
      - Soft Voting (equal-weight probability averaging)
      - Weighted Averaging (custom per-model weights)
      - Rank Averaging (rank-based fusion)
      - Per-category evaluation (seen/unseen lang × domain)
      - Weight optimisation on a labelled validation set
    """

    def __init__(self, task: str = "A", device: str = "cuda", max_length: int = 512):
        self.task = task
        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_length = max_length
        self.models: Dict[str, Any] = {}          # tag → (model, tokenizer)
        self.prob_cache: Dict[str, np.ndarray] = {}   # tag → (N, C) probs
        self._ids: Optional[np.ndarray] = None

    # ── Model Registration ─────────────────────────────────────────────

    def add_model(self, tag: str, trainer_obj):
        """
        Register a model from an existing trainer object (e.g. CodeClassifierTrainer,
        GraphCodeBERTTrainerA, etc.).

        Args:
            tag:          identifier for this model (e.g. "codebert")
            trainer_obj:  object with .model and .tokenizer attributes
        """
        model = trainer_obj.model
        tokenizer = trainer_obj.tokenizer
        model.to(self.device)
        model.eval()
        self.models[tag] = (model, tokenizer)
        logger.info(f"Registered model: {tag} ({model.config.num_labels} labels)")

    def add_model_from_path(self, tag: str, model_path: str):
        """
        Register a model by loading from a saved checkpoint directory.
        """
        from transformers import RobertaTokenizer, RobertaForSequenceClassification

        logger.info(f"Loading {tag} from {model_path} …")
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        model = RobertaForSequenceClassification.from_pretrained(model_path)
        model.to(self.device)
        model.eval()
        self.models[tag] = (model, tokenizer)
        logger.info(f"Registered model: {tag} ({model.config.num_labels} labels)")

    # ── Inference ──────────────────────────────────────────────────────

    @torch.no_grad()
    def _predict_single(
        self,
        tag: str,
        codes: List[str],
        batch_size: int = 32,
        code_features_fn=None,
    ) -> np.ndarray:
        """
        Run inference for a single model. Returns (N, C) softmax probabilities.

        Args:
            tag:              model tag
            codes:            list of code strings
            batch_size:       batch size for inference
            code_features_fn: optional callable(codes) → torch.Tensor for
                              feature-augmented models (e.g. CodeBERT+Features)
        """
        model, tokenizer = self.models[tag]
        model.to(self.device)
        model.eval()

        all_probs = []
        for i in tqdm(range(0, len(codes), batch_size), desc=f"  [{tag}]"):
            batch = codes[i: i + batch_size]
            enc = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            fwd_kwargs = dict(
                input_ids=enc["input_ids"].to(self.device),
                attention_mask=enc["attention_mask"].to(self.device),
            )

            # Support for feature-augmented models
            if code_features_fn is not None:
                fwd_kwargs["code_features"] = code_features_fn(batch).to(self.device)

            logits = model(**fwd_kwargs).logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

        return np.concatenate(all_probs, axis=0)

    def predict(
        self,
        parquet_path: str,
        strategy: str = "soft_vote",
        weights: Optional[List[float]] = None,
        batch_size: int = 32,
        code_features_fn_map: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Run ensemble prediction on a test parquet file.

        Args:
            parquet_path:          path to test .parquet file
            strategy:              'soft_vote', 'weighted_avg', or 'rank_avg'
            weights:               per-model weights for weighted_avg
            batch_size:            inference batch size
            code_features_fn_map:  dict of {tag: callable} for feature extraction

        Returns:
            DataFrame with columns: ID, prediction, confidence, 
                                    and per-model predictions
        """
        if not self.models:
            raise ValueError("No models registered. Use add_model() first.")

        # Load data
        df = pd.read_parquet(parquet_path)
        df = df.dropna(subset=["code"]).reset_index(drop=True)
        codes = df["code"].tolist()

        if "ID" in df.columns:
            ids = df["ID"].astype(str).values
        elif "id" in df.columns:
            ids = df["id"].astype(str).values
        else:
            ids = np.arange(len(df)).astype(str)

        self._ids = ids
        tags = list(self.models.keys())
        code_features_fn_map = code_features_fn_map or {}

        logger.info(f"Ensemble inference: {len(tags)} models × {len(codes)} samples")
        logger.info(f"Strategy: {strategy}")

        # ── Per-model inference ──
        prob_list = []
        for tag in tags:
            fn = code_features_fn_map.get(tag, None)
            probs = self._predict_single(tag, codes, batch_size, fn)
            self.prob_cache[tag] = probs
            prob_list.append(probs)
            logger.info(f"  [{tag}] shape={probs.shape}  "
                        f"mean_conf={probs.max(axis=1).mean():.4f}")

        # ── Combine ──
        if strategy == "soft_vote":
            combined = self._soft_vote(prob_list)
        elif strategy == "weighted_avg":
            if weights is None:
                weights = [1.0 / len(prob_list)] * len(prob_list)
            combined = self._weighted_avg(prob_list, weights)
        elif strategy == "rank_avg":
            combined = self._rank_avg(prob_list)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        predictions = combined.argmax(axis=1)
        confidence = combined.max(axis=1)

        # ── Agreement analysis ──
        per_model_preds = {tag: p.argmax(axis=1) for tag, p in zip(tags, prob_list)}
        preds_array = np.stack(list(per_model_preds.values()), axis=0)
        agreement = np.all(preds_array == preds_array[0:1], axis=0)
        logger.info(f"Model agreement: {agreement.mean():.2%} "
                    f"({agreement.sum()}/{len(codes)})")

        # ── Build result DataFrame ──
        result = pd.DataFrame({"ID": ids, "prediction": predictions.astype(int)})
        result["confidence"] = np.round(confidence, 4)
        result["agreement"] = agreement.astype(int)

        for tag in tags:
            result[f"{tag}_pred"] = per_model_preds[tag]

        logger.info(f"Mean confidence: {confidence.mean():.4f}")

        # Keep reference to full data for category evaluation
        self._last_full_df = df
        self._last_result = result

        return result

    # ── Ensemble strategies ────────────────────────────────────────────

    @staticmethod
    def _soft_vote(prob_list: List[np.ndarray]) -> np.ndarray:
        return np.mean(np.stack(prob_list, axis=0), axis=0)

    @staticmethod
    def _weighted_avg(prob_list: List[np.ndarray], weights: List[float]) -> np.ndarray:
        w = np.array(weights, dtype=np.float64)
        w /= w.sum()
        stacked = np.stack(prob_list, axis=0)
        return (stacked * w[:, None, None]).sum(axis=0)

    @staticmethod
    def _rank_avg(prob_list: List[np.ndarray]) -> np.ndarray:
        from scipy.stats import rankdata
        ranks = [np.apply_along_axis(rankdata, 1, p) for p in prob_list]
        avg = np.mean(np.stack(ranks, axis=0), axis=0)
        return avg / avg.sum(axis=1, keepdims=True)

    # ── Save Submission ───────────────────────────────────────────────

    def save_submission(self, result: pd.DataFrame, output_path: str):
        """Save a competition-ready submission CSV (ID, prediction)."""
        submission = result[["ID", "prediction"]].copy()
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        submission.to_csv(output_path, index=False)
        logger.info(f"Submission saved → {output_path}  ({len(submission)} rows)")

    # ── Evaluation ────────────────────────────────────────────────────

    def evaluate(
        self,
        result: pd.DataFrame,
        gold_path: Optional[str] = None,
        gold_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """
        Evaluate ensemble predictions. Provide either gold_path or gold_df.

        Prints overall metrics + per-category breakdown (Task A).
        """
        from sklearn.metrics import (
            f1_score, accuracy_score,
            precision_recall_fscore_support,
            classification_report,
        )

        if gold_df is None:
            if gold_path is None:
                # Try to use labels from the original data
                if hasattr(self, "_last_full_df") and "label" in self._last_full_df.columns:
                    gold_df = self._last_full_df
                else:
                    raise ValueError("Provide gold_path or gold_df for evaluation.")
            else:
                gold_df = pd.read_csv(gold_path)

        # Merge
        if "ID" in gold_df.columns and "ID" in result.columns:
            merged = pd.merge(
                gold_df, result[["ID", "prediction"]],
                on="ID", how="inner",
            )
        else:
            merged = gold_df.copy()
            merged["prediction"] = result["prediction"].values

        if "label" not in merged.columns:
            logger.warning("No 'label' column in gold data.")
            return {}

        y_true = merged["label"].values
        y_pred = merged["prediction"].values

        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        p, r, f1_w, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )

        print("\n" + "=" * 70)
        print(f"  ENSEMBLE EVALUATION — Task {self.task}")
        print("=" * 70)
        print(f"  Macro F1:       {macro_f1:.4f}   ← competition metric")
        print(f"  Accuracy:       {acc:.4f}")
        print(f"  Weighted F1:    {f1_w:.4f}")
        print(f"  Precision:      {p:.4f}")
        print(f"  Recall:         {r:.4f}")
        print()

        label_names = list(TASK_LABELS.get(self.task, {}).values())
        if label_names:
            print(classification_report(
                y_true, y_pred,
                target_names=label_names[:len(set(y_true))],
                zero_division=0,
            ))

        # ── Per-category breakdown ──
        self._evaluate_by_category(merged)

        print("=" * 70)

        # ── Per-model comparison ──
        tags = [c.replace("_pred", "") for c in result.columns if c.endswith("_pred")]
        if tags:
            print("\n  Per-Model Performance:")
            print("  " + "-" * 50)
            for tag in tags:
                col = f"{tag}_pred"
                if col in result.columns:
                    # Align with merged
                    if "ID" in result.columns and "ID" in merged.columns:
                        m2 = pd.merge(
                            merged[["ID", "label"]],
                            result[["ID", col]],
                            on="ID",
                        )
                    else:
                        m2 = merged.copy()
                        m2[col] = result[col].values

                    mf1 = f1_score(m2["label"], m2[col], average="macro", zero_division=0)
                    a = accuracy_score(m2["label"], m2[col])
                    print(f"  {tag:20s}  Macro F1={mf1:.4f}  Acc={a:.4f}")

            # Ensemble improvement
            print(f"  {'ENSEMBLE':20s}  Macro F1={macro_f1:.4f}  Acc={acc:.4f}")
            print("  " + "-" * 50)

        return {
            "macro_f1": macro_f1,
            "accuracy": acc,
            "weighted_f1": f1_w,
        }

    def _evaluate_by_category(self, df: pd.DataFrame):
        """Print per-category metrics for Task A (seen/unseen lang × domain)."""
        from sklearn.metrics import f1_score, accuracy_score

        lang_col = next(
            (c for c in df.columns if c.lower() in ("language", "lang", "programming_language")),
            None,
        )
        domain_col = next(
            (c for c in df.columns if c.lower() in ("domain", "task_type", "category")),
            None,
        )

        if lang_col is None or domain_col is None or "label" not in df.columns:
            return

        norm = lambda v: str(v).strip().lower()
        df = df.copy()
        df["_l"] = df[lang_col].apply(norm)
        df["_d"] = df[domain_col].apply(norm)

        settings = [
            ("(i)   Seen Lang & Seen Domain",    SEEN_LANGS,   SEEN_DOMAINS),
            ("(ii)  Unseen Lang & Seen Domain",   UNSEEN_LANGS, SEEN_DOMAINS),
            ("(iii) Seen Lang & Unseen Domain",   SEEN_LANGS,   UNSEEN_DOMAINS),
            ("(iv)  Unseen Lang & Unseen Domain", UNSEEN_LANGS, UNSEEN_DOMAINS),
        ]

        print("\n  Per-Category Breakdown:")
        for name, langs, domains in settings:
            mask = df["_l"].isin(langs) & df["_d"].isin(domains)
            sub = df[mask]
            n = len(sub)
            if n == 0:
                print(f"    {name}:  ** no samples **")
                continue
            mf1 = f1_score(sub["label"], sub["prediction"], average="macro", zero_division=0)
            acc = accuracy_score(sub["label"], sub["prediction"])
            print(f"    {name}  (n={n})  Macro F1={mf1:.4f}  Acc={acc:.4f}")

    # ── Weight Optimisation ───────────────────────────────────────────

    def optimize_weights(
        self,
        parquet_path: str,
        gold_path: Optional[str] = None,
        gold_df: Optional[pd.DataFrame] = None,
        batch_size: int = 32,
        n_steps: int = 21,
    ) -> Tuple[List[float], float]:
        """
        Search for optimal ensemble weights on a labelled validation set.

        1. If probabilities are not cached, runs inference first.
        2. Grid-searches over the weight simplex.
        3. Returns (best_weights, best_macro_f1).
        """
        from sklearn.metrics import f1_score

        # Ensure we have probabilities
        if not self.prob_cache:
            logger.info("No cached probabilities — running inference …")
            self.predict(parquet_path, strategy="soft_vote", batch_size=batch_size)

        # Load gold labels
        if gold_df is None:
            if gold_path is not None:
                gold_df = pd.read_csv(gold_path)
            elif hasattr(self, "_last_full_df") and "label" in self._last_full_df.columns:
                gold_df = self._last_full_df
            else:
                raise ValueError("Provide gold_path or gold_df for optimisation.")

        # Build y_true aligned with our prediction order
        ids = self._ids
        if "ID" in gold_df.columns:
            id_to_label = dict(zip(gold_df["ID"].astype(str), gold_df["label"]))
            y_true = np.array([id_to_label.get(str(i), -1) for i in ids])
        else:
            y_true = gold_df["label"].values

        valid = y_true >= 0
        y_true = y_true[valid]

        tags = list(self.prob_cache.keys())
        prob_list = [self.prob_cache[t][valid] for t in tags]
        n_models = len(prob_list)

        logger.info(f"Optimising weights for {tags} on {len(y_true)} samples …")

        best_f1 = -1.0
        best_w = None

        if n_models == 2:
            for w0 in np.linspace(0, 1, n_steps):
                w = [w0, 1.0 - w0]
                combined = self._weighted_avg(prob_list, w)
                preds = combined.argmax(axis=1)
                f1 = f1_score(y_true, preds, average="macro", zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_w = w

        elif n_models == 3:
            for w0 in np.linspace(0, 1, n_steps):
                for w1 in np.linspace(0, 1 - w0, max(2, int(n_steps * (1 - w0)))):
                    w2 = 1.0 - w0 - w1
                    if w2 < -1e-9:
                        continue
                    w = [w0, w1, max(0, w2)]
                    combined = self._weighted_avg(prob_list, w)
                    preds = combined.argmax(axis=1)
                    f1 = f1_score(y_true, preds, average="macro", zero_division=0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_w = w
        else:
            rng = np.random.default_rng(42)
            for _ in range(1000):
                w = rng.dirichlet(np.ones(n_models)).tolist()
                combined = self._weighted_avg(prob_list, w)
                preds = combined.argmax(axis=1)
                f1 = f1_score(y_true, preds, average="macro", zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_w = w

        w_str = ", ".join(f"{t}={w:.3f}" for t, w in zip(tags, best_w))
        logger.info(f"Optimal weights: {w_str}")
        logger.info(f"Optimal Macro F1: {best_f1:.4f}")

        return best_w, best_f1

    # ── Cleanup ───────────────────────────────────────────────────────

    def clear_gpu(self):
        """Free GPU memory for all registered models."""
        for tag, (model, _) in self.models.items():
            model.cpu()
            del model
        self.models.clear()
        self.prob_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("GPU memory cleared.")
