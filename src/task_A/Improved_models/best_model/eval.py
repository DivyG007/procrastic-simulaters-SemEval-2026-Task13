"""
Evaluation utilities for category-based reporting and metrics.
"""

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)

from data_utils import extract_code_features

SEEN_LANGS = {"c++", "cpp", "python", "java"}
UNSEEN_LANGS = {"go", "php", "c#", "csharp", "c", "javascript", "js"}
SEEN_DOMAINS = {"algorithmic"}
UNSEEN_DOMAINS = {"research", "production"}


def compute_metrics(eval_pred):
    """Compute weighted metrics for training calibration."""
    preds, labels = eval_pred
    preds = np.argmax(preds, axis=1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec}


def _norm(v):
    return str(v).strip().lower()


def evaluate_by_category(df, tag="Model"):
    """Print classification metrics for 4 evaluation settings + overall."""
    lang_col = next(
        (c for c in df.columns
         if c.lower() in ("language", "lang", "programming_language")),
        None,
    )
    domain_col = next(
        (c for c in df.columns
         if c.lower() in ("domain", "task_type", "category")),
        None,
    )

    if "label" not in df.columns:
        print(f"[{tag}] No 'label' column — cannot evaluate.")
        return

    if lang_col is None or domain_col is None:
        print(f"[{tag}] Missing language/domain columns. Overall only:")
        print(classification_report(df["label"], df["prediction"]))
        return

    df_eval = df.copy()
    df_eval["_l"] = df_eval[lang_col].apply(_norm)
    df_eval["_d"] = df_eval[domain_col].apply(_norm)

    settings = [
        ("(i)   Seen Lang & Seen Domain", SEEN_LANGS, SEEN_DOMAINS),
        ("(ii)  Unseen Lang & Seen Domain", UNSEEN_LANGS, SEEN_DOMAINS),
        ("(iii) Seen Lang & Unseen Domain", SEEN_LANGS, UNSEEN_DOMAINS),
        ("(iv)  Unseen Lang & Unseen Domain", UNSEEN_LANGS, UNSEEN_DOMAINS),
    ]

    print(f"\n{'=' * 70}")
    print(f"  TEST RESULTS — {tag}")
    print(f"{'=' * 70}")

    for name, langs, domains in settings:
        mask = df_eval["_l"].isin(langs) & df_eval["_d"].isin(domains)
        sub = df_eval[mask]
        n = len(sub)
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
    acc = accuracy_score(df_eval["label"], df_eval["prediction"])
    _, _, f1, _ = precision_recall_fscore_support(
        df_eval["label"], df_eval["prediction"], average="weighted", zero_division=0
    )
    print(f"\n  OVERALL  (n={len(df_eval)})  Accuracy={acc:.4f}  F1={f1:.4f}")
    print("=" * 70)


@torch.no_grad()
def predict_on_dataset(model, tokenizer, df, max_length=512, batch_size=32,
                       device="cuda"):
    """Run inference on a dataframe of code snippets."""
    model.to(device)
    model.eval()
    codes = df["code"].tolist()
    preds = []

    for i in tqdm(range(0, len(codes), batch_size), desc="Predicting"):
        batch = codes[i: i + batch_size]
        enc = tokenizer(
            batch, truncation=True, padding=True,
            max_length=max_length, return_tensors="pt",
        )
        fwd_kwargs = {
            "input_ids": enc["input_ids"].to(device),
            "attention_mask": enc["attention_mask"].to(device),
            "code_features": torch.tensor(
                [extract_code_features(c) for c in batch],
                dtype=torch.float32,
            ).to(device),
        }
        logits = model(**fwd_kwargs).logits
        preds.extend(logits.argmax(dim=-1).cpu().tolist())

    result = df.copy()
    result["prediction"] = preds
    return result
