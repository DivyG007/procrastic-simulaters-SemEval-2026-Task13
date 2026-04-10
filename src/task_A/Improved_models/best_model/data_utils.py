"""
Utility functions for data cleaning, feature extraction, and dataset preparation.
"""

import re
import math
import hashlib
import torch
import pandas as pd
from collections import Counter
from datasets import Dataset
from transformers import DataCollatorWithPadding

PLACEHOLDERS = {"unk", "na", "n/a", "-", "?", "none", "null", "nan"}
ENCODING_RE = re.compile(r"Ã.|â\u20ac™|â\u20acœ|â\u20ac|\ufffd|\?\?\?")


def normalize_ws(s: pd.Series) -> pd.Series:
    """Normalize whitespace in a pandas Series of strings."""
    return s.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def sha1_series(s: pd.Series) -> pd.Series:
    """Compute SHA-1 hash for each string in a pandas Series."""
    return s.map(lambda x: hashlib.sha1(x.encode("utf-8", "ignore")).hexdigest())


def clean_task_a(df: pd.DataFrame, text_col: str = "code",
                 label_col: str = "label") -> pd.DataFrame:
    """
    Apply cleaning steps to the dataset:
      1. Drop rows with missing / placeholder code
      2. Drop encoding-suspect rows
      3. Exact deduplication (SHA-1 of whitespace-normalised text)
      4. Drop contradictory-label groups (same code → different labels)
    """
    n0 = len(df)

    # --- Drop missing-like rows ---
    raw = df[text_col]
    is_null = raw.isna()
    is_empty = raw.fillna("").astype(str).eq("")
    is_ws = raw.fillna("").astype(str).str.fullmatch(r"\s*")
    is_placeholder = raw.fillna("").astype(str).str.strip().str.lower().isin(PLACEHOLDERS)
    missing_mask = is_null | is_empty | is_ws | is_placeholder
    df = df[~missing_mask].reset_index(drop=True)
    print(f"  Dropped {missing_mask.sum()} missing/placeholder rows")

    # --- Drop encoding-suspect rows ---
    text_norm = normalize_ws(df[text_col])
    enc_mask = text_norm.str.contains(ENCODING_RE)
    n_enc = enc_mask.sum()
    df = df[~enc_mask].reset_index(drop=True)
    print(f"  Dropped {n_enc} encoding-suspect rows")

    # --- Exact deduplication ---
    text_norm = normalize_ws(df[text_col])
    exact_hash = sha1_series(text_norm)
    dup_mask = exact_hash.duplicated(keep="first")
    n_dup = dup_mask.sum()
    df = df[~dup_mask].reset_index(drop=True)
    print(f"  Dropped {n_dup} exact duplicates")

    # --- Drop contradictory-label groups ---
    text_norm = normalize_ws(df[text_col])
    exact_hash = sha1_series(text_norm)
    grp = pd.DataFrame({"h": exact_hash, "y": df[label_col].astype(str)})
    bad_hashes = set(grp.groupby("h")["y"].nunique().pipe(lambda s: s[s > 1]).index)
    contra_mask = exact_hash.isin(bad_hashes)
    n_contra = contra_mask.sum()
    df = df[~contra_mask].reset_index(drop=True)
    print(f"  Dropped {n_contra} contradictory-label rows")

    print(f"  Cleaning: {n0} \u2192 {len(df)} rows")
    return df


def extract_code_features(code: str) -> list:
    """
    Returns 8 handcrafted float values in roughly [0, 1]:
      1. comment_ratio
      2. avg_line_length
      3. line_length_cv
      4. whitespace_consistency
      5. max_nesting_depth
      6. blank_line_ratio
      7. char_entropy (Shannon)
      8. bracket_density
    """
    lines = code.split("\n")
    non_empty = [l for l in lines if l.strip()]
    total_lines = max(len(lines), 1)
    ne_count = max(len(non_empty), 1)

    # 1. Comment ratio
    comment_lines = sum(
        1 for l in lines
        if l.strip().startswith("#") or l.strip().startswith("//")
    )
    comment_ratio = comment_lines / total_lines

    # 2. Avg line length
    lengths = [len(l) for l in non_empty]
    mean_len = sum(lengths) / ne_count if lengths else 0.0
    avg_line_len = min(mean_len / 200.0, 1.0)

    # 3. Line length CV
    if lengths and mean_len > 0:
        var = sum((x - mean_len) ** 2 for x in lengths) / ne_count
        cv = math.sqrt(var) / mean_len
    else:
        cv = 0.0
    line_len_cv = min(cv / 2.0, 1.0)

    # 4. Whitespace consistency
    leading = [len(l) - len(l.lstrip()) for l in non_empty]
    if leading:
        m = sum(leading) / len(leading)
        ws_var = sum((x - m) ** 2 for x in leading) / len(leading)
        ws_std = math.sqrt(ws_var)
    else:
        ws_std = 0.0
    ws_consistency = min(ws_std / 20.0, 1.0)

    # 5. Max nesting depth (proxy via leading WS)
    max_indent = max(leading) if leading else 0
    max_depth = min(max_indent / 32.0, 1.0)

    # 6. Blank line ratio
    blank_lines = sum(1 for l in lines if not l.strip())
    blank_line_ratio = blank_lines / total_lines

    # 7. Character entropy
    if len(code) > 0:
        char_counts = Counter(code)
        total_chars = len(code)
        entropy = -sum(
            (c / total_chars) * math.log2(c / total_chars)
            for c in char_counts.values()
        )
        char_entropy = min(entropy / 7.0, 1.0)
    else:
        char_entropy = 0.0

    # 8. Bracket density
    brackets = sum(1 for c in code if c in "()[]{}")
    bracket_density = min(brackets / max(len(code), 1) * 20.0, 1.0)

    return [
        comment_ratio, avg_line_len, line_len_cv,
        ws_consistency, max_depth,
        blank_line_ratio, char_entropy, bracket_density,
    ]


class FeaturesDataCollator:
    """
    Data collator that wraps DataCollatorWithPadding and additionally
    stacks 'code_features' into the batch.
    """
    def __init__(self, tokenizer):
        self.base = DataCollatorWithPadding(tokenizer=tokenizer)

    def __call__(self, features):
        code_feats = None
        if "code_features" in features[0]:
            code_feats = [f.pop("code_features") for f in features]
        batch = self.base(features)
        if code_feats is not None:
            batch["code_features"] = torch.tensor(
                code_feats, dtype=torch.float32
            )
        return batch


def make_hf_dataset(df: pd.DataFrame, tokenizer, max_length=512) -> Dataset:
    """
    Convert a pandas DataFrame into a HuggingFace Dataset with tokenization
    and handcrafted features.
    """
    def tokenize_fn(examples):
        return tokenizer(
            examples["code"],
            truncation=True,
            max_length=max_length,
        )

    def add_features(examples):
        return {"code_features": [extract_code_features(c) for c in examples["code"]]}

    ds = Dataset.from_pandas(df[["code", "label"]].reset_index(drop=True))
    ds = ds.map(add_features, batched=True, desc="Extracting features")
    ds = ds.map(tokenize_fn, batched=True, remove_columns=["code"],
                desc="Tokenising")
    ds = ds.rename_column("label", "labels")
    return ds
