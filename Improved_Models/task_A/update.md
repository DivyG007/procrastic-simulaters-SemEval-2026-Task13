# GraphCodeBERT Model Update Log (Task A)

This document tracks the features, architectural enhancements, and training methods implemented in the `graphcodebert_task_a.ipynb` notebook to tackle SemEval-2026 Task 13 (Machine-Generated Code Detection).

## 🚀 Successfully Implemented Features

### 1. Extended Sequence Context (via Gradient Checkpointing)
* **What**: Increased `MAX_LENGTH` from 256 to `512` and enabled `gradient_checkpointing=True` in `TrainingArguments`.
* **Why**: Algorithmic code (like LeetCode solutions) is frequently longer than 256 tokens. Truncating heavily loses structural information. Gradient checkpointing sacrifices a small amount of training speed to massively reduce GPU memory footprint, allowing 512-length sequences to comfortably train on 16GB Kaggle T4 GPUs.

### 2. Cross-Lingual Robustness via LOLO Validation
* **What**: Configured a Leave-One-Language-Out (LOLO) validation strategy (controlled by `USE_LOLO_VALIDATION` and `HOLDOUT_LANGUAGE` flags).
* **Why**: Subtask A tests on *unseen* languages (like Go, PHP, C#). Random, stratified validation splits across all seen languages (C++, Python, Java) produce an artificially inflated F1 score. LOLO validation (e.g., train on Python/Java, validate on C++) gives a much more authentic assessment of how language-agnostic the model features are.

### 3. Hard-Example Mining (Focal Loss)
* **What**: Replaced the standard Cross-Entropy loss with a custom `FocalLossTrainer` ($\alpha=1.0$, $\gamma=2.0$).
* **Why**: While the dataset is well-balanced, it contains many "easy" samples (terrible LLM code vs extremely idiosyncratic human code) and a subset of very "hard" samples (highly optimized human code vs cutting-edge LLM output). Focal Loss actively down-weights the easily classified examples to force the gradients to solve the difficult borderline boundaries.

### 4. Multi-Sample Dropout Architecture
* **What**: Swapped the standard `RobertaForSequenceClassification` wrapper for a custom `GraphCodeBERTMultiDropModel`.
* **Why**: A standard linear classification head is prone to overfitting and instability during fine-tuning on massive code corpora. This custom head extracts the `[CLS]` embedding, splits it through multiple parallel dropout layers with sequentially increasing drop rates, and averages the output logits. This acts as an ensemble-within-a-layer, heavily stabilizing validation metrics and avoiding aggressive over-confidence.

---

## ⏸️ Deferred Features
### 1. Dynamic Data Flow Graph (DFG) Extraction
* **Status**: Paused / Excluded from Notebook preprocessing.
* **Reason**: True GraphCodeBERT performance requires injecting AST-derived Data Flow Graph variables into the sequence. Doing this dynamically inside a Kaggle notebook via `tree-sitter` for 500,000 algorithmic snippets triggers CPU timeouts.
* **Resolution**: DFG extraction must be shifted into an **offline dataset preparation pipeline**. The resulting DFG mappings (`position_ids`, `attention_mask`) should be saved explicitly into `.parquet` chunks and uploaded directly to Kaggle.