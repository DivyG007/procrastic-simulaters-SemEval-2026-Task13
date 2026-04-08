"""
train_utils.py — Training Utilities
=====================================
Shared training infrastructure used by all three per-backbone scripts:
  - FeaturesDataCollator   — stacks code_features alongside tokenised batch
  - get_layer_wise_lr_groups — LLRD parameter grouping
  - DeepHeadTrainer        — HF Trainer subclass with LLRD + cosine scheduler
  - compute_metrics        — accuracy, F1, precision, recall
  - evaluate_by_category   — 4-category evaluation breakdown
  - predict_on_dataset     — batch inference returning predictions + probs
  - save_probabilities     — serialise softmax probs to .npy for ensemble
"""

import os
import gc
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    Trainer,
    DataCollatorWithPadding,
    get_cosine_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)

from config import (
    SEEN_LANGS,
    UNSEEN_LANGS,
    SEEN_DOMAINS,
    UNSEEN_DOMAINS,
    MAX_LENGTH,
    ENSEMBLE_CACHE_DIR,
)
from data_utils import extract_code_features


# =====================================================================
#  Data Collator
# =====================================================================

class FeaturesDataCollator:
    """
    Wraps DataCollatorWithPadding to additionally stack the
    `code_features` tensor alongside the standard tokenised batch.
    """

    def __init__(self, tokenizer):
        self.base = DataCollatorWithPadding(tokenizer=tokenizer)

    def __call__(self, features):
        # Pop code_features before the base collator processes the batch
        code_feats = None
        if "code_features" in features[0]:
            code_feats = [f.pop("code_features") for f in features]

        batch = self.base(features)

        if code_feats is not None:
            batch["code_features"] = torch.tensor(code_feats, dtype=torch.float32)

        return batch


# =====================================================================
#  Layer-Wise Learning-Rate Decay (LLRD)
# =====================================================================

def get_layer_wise_lr_groups(
    model,
    base_lr: float = 2e-5,
    head_lr: float = 1e-3,
    weight_decay: float = 0.01,
    llrd_factor: float = 0.95,
):
    """
    Build parameter groups with per-layer learning rates:
      - Embedding layer   → base_lr × llrd_factor^N  (lowest LR)
      - Encoder layer i   → base_lr × llrd_factor^(N-i)
      - Classification head → head_lr                (highest LR)

    Bias and LayerNorm parameters get weight_decay = 0.
    """
    opt_params = []
    no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}

    num_layers = model.config.num_hidden_layers  # 12 for all three backbones

    # ── Embedding parameters ──
    emb_params_wd, emb_params_nowd = [], []
    for n, p in model.transformer.embeddings.named_parameters():
        if any(nd in n for nd in no_decay):
            emb_params_nowd.append(p)
        else:
            emb_params_wd.append(p)

    emb_lr = base_lr * (llrd_factor ** num_layers)
    if emb_params_wd:
        opt_params.append({"params": emb_params_wd, "lr": emb_lr, "weight_decay": weight_decay})
    if emb_params_nowd:
        opt_params.append({"params": emb_params_nowd, "lr": emb_lr, "weight_decay": 0.0})

    # ── Encoder layers ──
    for i in range(num_layers):
        layer    = model.transformer.encoder.layer[i]
        layer_lr = base_lr * (llrd_factor ** (num_layers - i))
        wd_p, nowd_p = [], []
        for n, p in layer.named_parameters():
            if any(nd in n for nd in no_decay):
                nowd_p.append(p)
            else:
                wd_p.append(p)
        if wd_p:
            opt_params.append({"params": wd_p, "lr": layer_lr, "weight_decay": weight_decay})
        if nowd_p:
            opt_params.append({"params": nowd_p, "lr": layer_lr, "weight_decay": 0.0})

    # ── Classification head + feature norm ──
    head_wd, head_nowd = [], []
    for module in [model.head, model.feat_norm]:
        for n, p in module.named_parameters():
            if any(nd in n for nd in no_decay):
                head_nowd.append(p)
            else:
                head_wd.append(p)

    if head_wd:
        opt_params.append({"params": head_wd, "lr": head_lr, "weight_decay": weight_decay})
    if head_nowd:
        opt_params.append({"params": head_nowd, "lr": head_lr, "weight_decay": 0.0})

    return opt_params


# =====================================================================
#  Custom Trainer
# =====================================================================

class DeepHeadTrainer(Trainer):
    """
    HuggingFace Trainer subclass that injects:
      - Layer-wise learning-rate decay (LLRD)
      - Cosine LR schedule with linear warmup
    """

    def __init__(self, *args, llrd_factor=0.95, head_lr=1e-3, **kwargs):
        self.llrd_factor = llrd_factor
        self.head_lr     = head_lr
        super().__init__(*args, **kwargs)

    def create_optimizer_and_scheduler(self, num_training_steps):
        """Override to use LLRD param groups + cosine schedule."""
        param_groups = get_layer_wise_lr_groups(
            self.model,
            base_lr=self.args.learning_rate,
            head_lr=self.head_lr,
            weight_decay=self.args.weight_decay,
            llrd_factor=self.llrd_factor,
        )
        self.optimizer = torch.optim.AdamW(param_groups)

        warmup_steps = int(num_training_steps * self.args.warmup_ratio)
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )


# =====================================================================
#  Metrics
# =====================================================================

def compute_metrics(eval_pred):
    """Compute accuracy, F1, precision, recall for the HF Trainer."""
    preds, labels = eval_pred
    preds = np.argmax(preds, axis=1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec}


# =====================================================================
#  4-Category Evaluation Breakdown
# =====================================================================

def _norm(v):
    """Normalise a string for category matching."""
    return str(v).strip().lower()


def evaluate_by_category(df: pd.DataFrame, tag: str = "Model"):
    """
    Print classification metrics for 4 evaluation settings + overall:
      (i)   Seen Language   & Seen Domain
      (ii)  Unseen Language & Seen Domain
      (iii) Seen Language   & Unseen Domain
      (iv)  Unseen Language & Unseen Domain
    """
    lang_col = next(
        (c for c in df.columns if c.lower() in ("language", "lang", "programming_language")),
        None,
    )
    domain_col = next(
        (c for c in df.columns if c.lower() in ("domain", "task_type", "category")),
        None,
    )

    if "label" not in df.columns:
        print(f"[{tag}] No 'label' column — cannot evaluate.")
        return

    if lang_col is None or domain_col is None:
        print(f"[{tag}] Missing language/domain columns. Overall only:")
        print(classification_report(df["label"], df["prediction"]))
        return

    df["_l"] = df[lang_col].apply(_norm)
    df["_d"] = df[domain_col].apply(_norm)

    settings = [
        ("(i)   Seen Lang & Seen Domain",     SEEN_LANGS,   SEEN_DOMAINS),
        ("(ii)  Unseen Lang & Seen Domain",   UNSEEN_LANGS, SEEN_DOMAINS),
        ("(iii) Seen Lang & Unseen Domain",   SEEN_LANGS,   UNSEEN_DOMAINS),
        ("(iv)  Unseen Lang & Unseen Domain", UNSEEN_LANGS, UNSEEN_DOMAINS),
    ]

    print(f"\n{'=' * 70}")
    print(f"  TEST RESULTS — {tag}")
    print(f"{'=' * 70}")

    for name, langs, domains in settings:
        mask = df["_l"].isin(langs) & df["_d"].isin(domains)
        sub  = df[mask]
        n    = len(sub)
        if n == 0:
            print(f"\n  {name}:  ** no samples **")
            continue
        y_t, y_p = sub["label"].values, sub["prediction"].values
        acc = accuracy_score(y_t, y_p)
        p, r, f1, _ = precision_recall_fscore_support(
            y_t, y_p, average="weighted", zero_division=0
        )
        print(f"\n  {name}  (n={n})")
        print(f"    Accuracy={acc:.4f}  Prec={p:.4f}  Recall={r:.4f}  F1={f1:.4f}")
        print(classification_report(y_t, y_p, zero_division=0))

    # Overall
    acc = accuracy_score(df["label"], df["prediction"])
    _, _, f1, _ = precision_recall_fscore_support(
        df["label"], df["prediction"], average="weighted", zero_division=0
    )
    print(f"\n  OVERALL  (n={len(df)})  Accuracy={acc:.4f}  F1={f1:.4f}")
    print("=" * 70)

    df.drop(columns=["_l", "_d"], inplace=True, errors="ignore")


# =====================================================================
#  Inference  (predictions + softmax probabilities)
# =====================================================================

@torch.no_grad()
def predict_on_dataset(
    model,
    tokenizer,
    df: pd.DataFrame,
    max_length: int = MAX_LENGTH,
    batch_size: int = 32,
    device: str = "cuda",
):
    """
    Run batch inference on a DataFrame with a 'code' column.

    Returns:
        result_df:  copy of `df` with added 'prediction' column
        all_probs:  np.ndarray of shape (N, num_labels) softmax probabilities
    """
    model.to(device)
    model.eval()
    codes = df["code"].tolist()
    preds     = []
    all_probs = []

    for i in tqdm(range(0, len(codes), batch_size), desc="Predicting"):
        batch = codes[i : i + batch_size]

        enc = tokenizer(
            batch, truncation=True, padding=True,
            max_length=max_length, return_tensors="pt",
        )

        fwd_kwargs = {
            "input_ids":      enc["input_ids"].to(device),
            "attention_mask":  enc["attention_mask"].to(device),
            "code_features":   torch.tensor(
                [extract_code_features(c) for c in batch],
                dtype=torch.float32,
            ).to(device),
        }

        logits = model(**fwd_kwargs).logits
        probs  = F.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
        preds.extend(logits.argmax(dim=-1).cpu().tolist())

    result = df.copy()
    result["prediction"] = preds
    all_probs = np.concatenate(all_probs, axis=0)

    return result, all_probs


# =====================================================================
#  Save / Load Probabilities for Ensemble
# =====================================================================

def save_probabilities(
    probs: np.ndarray,
    model_tag: str,
    output_dir: str = ENSEMBLE_CACHE_DIR,
    ids: Optional[np.ndarray] = None,
):
    """
    Cache softmax probabilities as .npy for consumption by the ensemble.

    Saves:
      - {output_dir}/{model_tag}_probs.npy   (N, num_labels)
      - {output_dir}/sample_ids.npy           (N,)  — shared across models
    """
    os.makedirs(output_dir, exist_ok=True)

    prob_path = os.path.join(output_dir, f"{model_tag}_probs.npy")
    np.save(prob_path, probs)
    print(f"[{model_tag}] Saved probabilities → {prob_path}  shape={probs.shape}")

    if ids is not None:
        id_path = os.path.join(output_dir, "sample_ids.npy")
        np.save(id_path, ids)
        print(f"[{model_tag}] Saved sample IDs → {id_path}")


def cleanup_gpu():
    """Free GPU memory between model runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
