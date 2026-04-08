"""
data_utils.py — Data Loading, Cleaning & Feature Extraction
=============================================================
Mirrors the inline preprocessing from codebert_deep_head.ipynb:
  1. Drop missing / placeholder rows
  2. Flag & remove encoding-suspect rows
  3. Exact deduplication via SHA-1 of whitespace-normalised text
  4. Remove contradictory-label duplicates
  5. Stratified sampling + 80/10/10 split
  6. 8 language-agnostic stylometric features
  7. HuggingFace Dataset construction (tokenise + attach features)
"""

import re
import math
import hashlib
from collections import Counter
from typing import Tuple

import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split

from config import (
    TRAIN_PARQUET,
    SAMPLE_SIZE,
    RANDOM_SEED,
    TEST_RATIO,
    VAL_TEST_RATIO,
    MAX_LENGTH,
    NUM_CODE_FEATURES,
)

# =====================================================================
#  Cleaning Helpers
# =====================================================================

# Strings that should be treated as missing values
PLACEHOLDERS = {"unk", "na", "n/a", "-", "?", "none", "null", "nan"}

# Regex that flags common encoding-corruption artefacts
ENCODING_RE = re.compile(r"Ã.|â\u20ac™|â\u20acœ|â\u20ac|\ufffd|\?\?\?")


def _normalize_ws(s: pd.Series) -> pd.Series:
    """Collapse all runs of whitespace to a single space and strip."""
    return s.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def _sha1_series(s: pd.Series) -> pd.Series:
    """Compute per-row SHA-1 hashes for deduplication."""
    return s.map(lambda x: hashlib.sha1(x.encode("utf-8", "ignore")).hexdigest())


def clean_task_a(
    df: pd.DataFrame,
    text_col: str = "code",
    label_col: str = "label",
) -> pd.DataFrame:
    """
    Apply the same cleaning pipeline as clean_and_flag_task_a.py:
      1. Drop rows with missing / placeholder code
      2. Drop encoding-suspect rows
      3. Exact deduplication (SHA-1 of whitespace-normalised text)
      4. Drop contradictory-label groups (same code → different labels)

    Returns:
        Cleaned DataFrame.
    """
    n0 = len(df)

    # 1 — Drop missing-like rows
    raw = df[text_col]
    is_null        = raw.isna()
    is_empty       = raw.fillna("").astype(str).eq("")
    is_ws          = raw.fillna("").astype(str).str.fullmatch(r"\s*")
    is_placeholder = raw.fillna("").astype(str).str.strip().str.lower().isin(PLACEHOLDERS)
    missing_mask   = is_null | is_empty | is_ws | is_placeholder
    df = df[~missing_mask].reset_index(drop=True)
    print(f"  Dropped {missing_mask.sum()} missing/placeholder rows")

    # 2 — Drop encoding-suspect rows
    text_norm = _normalize_ws(df[text_col])
    enc_mask  = text_norm.str.contains(ENCODING_RE)
    n_enc     = enc_mask.sum()
    df = df[~enc_mask].reset_index(drop=True)
    print(f"  Dropped {n_enc} encoding-suspect rows")

    # 3 — Exact deduplication
    text_norm  = _normalize_ws(df[text_col])
    exact_hash = _sha1_series(text_norm)
    dup_mask   = exact_hash.duplicated(keep="first")
    n_dup      = dup_mask.sum()
    df = df[~dup_mask].reset_index(drop=True)
    print(f"  Dropped {n_dup} exact duplicates")

    # 4 — Drop contradictory-label groups
    text_norm  = _normalize_ws(df[text_col])
    exact_hash = _sha1_series(text_norm)
    grp = pd.DataFrame({"h": exact_hash, "y": df[label_col].astype(str)})
    bad_hashes  = set(grp.groupby("h")["y"].nunique().pipe(lambda s: s[s > 1]).index)
    contra_mask = exact_hash.isin(bad_hashes)
    n_contra    = contra_mask.sum()
    df = df[~contra_mask].reset_index(drop=True)
    print(f"  Dropped {n_contra} contradictory-label rows")

    print(f"  Cleaning: {n0} → {len(df)} rows")
    return df


# =====================================================================
#  Load, Clean, Sample, Split
# =====================================================================

def load_and_prepare_data(
    parquet_path: str = TRAIN_PARQUET,
    sample_size: int = SAMPLE_SIZE,
    seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    End-to-end data preparation:
      1. Read parquet
      2. Clean (dedup, encoding, contradictions)
      3. Stratified subsample to `sample_size`
      4. 80 / 10 / 10 stratified split

    Returns:
        (train_df, val_df, test_df)
    """
    print(f"Loading data from {parquet_path} ...")
    raw_df = pd.read_parquet(parquet_path)
    print(f"Raw dataset: {len(raw_df)} rows, columns: {raw_df.columns.tolist()}")
    raw_df["label"] = raw_df["label"].astype(int)

    # Clean
    df_clean = clean_task_a(raw_df, text_col="code", label_col="label")

    # Stratified subsample
    if sample_size < len(df_clean):
        df_sampled = (
            df_clean.groupby("label", group_keys=False)
            .apply(
                lambda x: x.sample(
                    n=max(1, int(sample_size * len(x) / len(df_clean))),
                    random_state=seed,
                )
            )
            .reset_index(drop=True)
        )
    else:
        df_sampled = df_clean.copy()

    print(f"Sampled: {len(df_sampled)} rows")
    print(df_sampled["label"].value_counts().sort_index())

    # 80 / 10 / 10 stratified split
    train_df, temp_df = train_test_split(
        df_sampled,
        test_size=TEST_RATIO,
        stratify=df_sampled["label"],
        random_state=seed,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=VAL_TEST_RATIO,
        stratify=temp_df["label"],
        random_state=seed,
    )

    print(f"\nSplits — Train: {len(train_df)},  Val: {len(val_df)},  Test: {len(test_df)}")
    print(f"Train labels:\n{train_df['label'].value_counts().sort_index()}")
    print(f"Val labels:\n{val_df['label'].value_counts().sort_index()}")
    print(f"Test labels:\n{test_df['label'].value_counts().sort_index()}")

    return train_df, val_df, test_df


# =====================================================================
#  Stylometric Feature Extraction  (8 language-agnostic signals)
# =====================================================================

def extract_code_features(code: str) -> list:
    """
    Extract 8 normalised stylometric features from a code snippet.

    Returns a list of 8 floats in roughly [0, 1]:
      1. comment_ratio        — fraction of lines that are comments
      2. avg_line_length      — mean non-empty line length (/ 200)
      3. line_length_cv       — coefficient of variation of line lengths
      4. whitespace_consistency — std of leading whitespace (/ 20)
      5. max_nesting_depth    — max indentation depth (/ 32)
      6. blank_line_ratio     — fraction of blank lines
      7. char_entropy         — Shannon entropy of characters (/ 7)
      8. bracket_density      — density of ()[]{}  characters
    """
    lines     = code.split("\n")
    non_empty = [l for l in lines if l.strip()]
    total_lines = max(len(lines), 1)
    ne_count    = max(len(non_empty), 1)

    # 1 — Comment ratio
    comment_lines = sum(
        1 for l in lines
        if l.strip().startswith("#") or l.strip().startswith("//")
    )
    comment_ratio = comment_lines / total_lines

    # 2 — Average line length (normalised)
    lengths  = [len(l) for l in non_empty]
    mean_len = sum(lengths) / ne_count if lengths else 0.0
    avg_line_len = min(mean_len / 200.0, 1.0)

    # 3 — Line-length coefficient of variation
    if lengths and mean_len > 0:
        var = sum((x - mean_len) ** 2 for x in lengths) / ne_count
        cv  = math.sqrt(var) / mean_len
    else:
        cv = 0.0
    line_len_cv = min(cv / 2.0, 1.0)

    # 4 — Whitespace consistency (leading-indent std)
    leading = [len(l) - len(l.lstrip()) for l in non_empty]
    if leading:
        m      = sum(leading) / len(leading)
        ws_var = sum((x - m) ** 2 for x in leading) / len(leading)
        ws_std = math.sqrt(ws_var)
    else:
        ws_std = 0.0
    ws_consistency = min(ws_std / 20.0, 1.0)

    # 5 — Max nesting depth (via indentation)
    max_indent = max(leading) if leading else 0
    max_depth  = min(max_indent / 32.0, 1.0)

    # 6 — Blank-line ratio
    blank_lines     = sum(1 for l in lines if not l.strip())
    blank_line_ratio = blank_lines / total_lines

    # 7 — Character entropy (Shannon, normalised to [0, 1])
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

    # 8 — Bracket density
    brackets        = sum(1 for c in code if c in "()[]{}")
    bracket_density = min(brackets / max(len(code), 1) * 20.0, 1.0)

    return [
        comment_ratio, avg_line_len, line_len_cv,
        ws_consistency, max_depth,
        blank_line_ratio, char_entropy, bracket_density,
    ]


# =====================================================================
#  HuggingFace Dataset Construction
# =====================================================================

def make_hf_dataset(df: pd.DataFrame, tokenizer, max_length: int = MAX_LENGTH) -> Dataset:
    """
    Convert a pandas DataFrame with 'code' and 'label' columns into a
    HuggingFace Dataset ready for the Trainer:
      - Extracts stylometric features
      - Tokenises code
      - Renames 'label' → 'labels' (HF convention)
    """
    ds = Dataset.from_pandas(df[["code", "label"]].reset_index(drop=True))

    # Attach stylometric features
    def add_features(examples):
        return {"code_features": [extract_code_features(c) for c in examples["code"]]}

    ds = ds.map(add_features, batched=True, desc="Extracting features")

    # Tokenise
    def tokenize_fn(examples):
        return tokenizer(examples["code"], truncation=True, max_length=max_length)

    ds = ds.map(tokenize_fn, batched=True, remove_columns=["code"], desc="Tokenising")

    # Rename label → labels (expected by HF Trainer)
    ds = ds.rename_column("label", "labels")

    return ds
