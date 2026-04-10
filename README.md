# SemEval-2026 Task 13 — Procrastic Simulators

This repository contains our experiments for detecting machine-generated code across:
- **Task A**: Human vs Machine (binary)
- **Task B**: 11-way authorship attribution (Human + 10 LLM families)
- **Task C**: Human / Machine / Hybrid / Adversarial

---

## Unified Interpretation Summary (Tasks A, B, C)

### Shared takeaways
- **In-distribution validation can be overly optimistic**; generalization is the real bottleneck.
- **Macro F1** is the reliable metric across tasks with class imbalance or distribution shift.
- Stronger results consistently came from:
  - better imbalance handling (class weights, focal loss)
  - richer heads (deep heads / BiLSTM / contrastive branches)
  - robust optimization (LLRD, cosine schedule, calibrated thresholds)

### Task A (binary: Human vs Machine)
- Baseline transformer models often reached **very high validation scores** but degraded on harder evaluation settings (unseen language/domain combinations).
- Better-performing A variants emphasized:
  - deeper classifier heads,
  - stylometric/code-structure features,
  - stronger regularization and loss shaping,
  - and cleaner data pipelines.
- Core interpretation: Task A is not just separability; it is **cross-distribution robustness**.

### Task B (11-way authorship attribution)
- Main challenge is **extreme imbalance** (human class dominates heavily).
- Vanilla cross-entropy setups were biased toward majority predictions.
- Stronger B variants improved minority-family recall using:
  - balanced/weighted losses,
  - focal + supervised contrastive objectives,
  - GraphCodeBERT/UniXcoder-style representations,
  - layer-wise LR decay,
  - and per-class threshold calibration.
- Core interpretation: most gains came from **minority-class recoverability**, not raw accuracy.

### Task C (4-way: Human/Machine/Hybrid/Adversarial)
- Hardest classes are typically **Hybrid** and **Adversarial** because boundaries are subtle.
- Baseline models were strongest on pure Human/Machine and weaker on mixed/deceptive patterns.
- Improved approaches targeted this by combining:
  - stronger backbones,
  - sequence-aware heads (e.g., BiLSTM or deeper fusion heads),
  - class-imbalance-aware losses,
  - and contrastive separation.
- Core interpretation: Task C requires modeling **fine-grained mixture signals**, not just source identity.

---

## How to Run (Best-Model Python Pipelines, Local)

> This section is intentionally only for the **pythonic best_model pipelines**.

### 1) Task A best_model
Location: `src/task_A/Improved_models/best_model`

What to set locally:
- In `src/task_A/Improved_models/best_model/config.py`
  - set `TRAIN_PARQUET` to your local train parquet path
  - set `TEST_PARQUET` to your local test parquet path (optional)
  - set `OUTPUT_DIR` to a local writable folder

Run:
1. `cd src/task_A/Improved_models/best_model`
2. `python3 main.py`

Notes:
- This pipeline does its own split/eval flow internally.
- If `TEST_PARQUET` exists, it also runs official-test evaluation utilities.

### 2) Task B best_model
Location: `src/task_B/Improved_models/best_model`

Install dependencies:
- `pip install -r src/task_B/Improved_models/best_model/requirements.txt`

Local data options:
- Option A (recommended): place local parquet files and set in config
  - `train_path`, `val_path`, `test_path` in `Config` (`config.py`)
- Option B: keep defaults; pipeline auto-downloads from HuggingFace if local files are missing

Important local tweak:
- In `src/task_B/Improved_models/best_model/config.py`, change `output_dir` from Kaggle-style to a local path (for example: `./results_graphcodebert_taskB`).

Run:
1. `cd src/task_B/Improved_models/best_model`
2. `python3 main.py`

Notes:
- Pipeline includes training, validation evaluation, threshold calibration, and submission generation.

---

## Minimal Project Pointers
- Official scorer: `scorer.py`
- Format checks: `format_checker.py`
- Label maps: `dataset_format/task_A`, `dataset_format/task_B`, `dataset_format/task_C`
- Baselines: `baselines/`
- Improved experiments: `src/task_A/Improved_models`, `src/task_B/Improved_models`, `src/task_C/Improved_models`

---

## Notebook Map (Name → Key Features → Purpose)

### Task A

| Notebook | Model key features / innovations | Purpose |
|---|---|---|
| [src/task_A/baseline/Task-A-Starter.ipynb](src/task_A/baseline/Task-A-Starter.ipynb) | Starter transformer pipeline | Quick sanity-check baseline |
| [src/task_A/baseline/Task-A-Baseline-limited.ipynb](src/task_A/baseline/Task-A-Baseline-limited.ipynb) | Limited-data baseline setup | Fast iteration baseline |
| [src/task_A/baseline/Task-A-Baseline-Full-Datset.ipynb](src/task_A/baseline/Task-A-Baseline-Full-Datset.ipynb) | Full-data baseline training | Reference full-data baseline |
| [src/task_A/Improved_models/codebert_linear_with_features.ipynb](src/task_A/Improved_models/codebert_linear_with_features.ipynb) | CodeBERT + linear head + handcrafted features | Feature-fusion ablation |
| [src/task_A/Improved_models/codebert_deep_head_with_features.ipynb](src/task_A/Improved_models/codebert_deep_head_with_features.ipynb) | CodeBERT + deep MLP head + features | Stronger feature-fusion variant |
| [src/task_A/Improved_models/codebert_deephead_14_features.ipynb](src/task_A/Improved_models/codebert_deephead_14_features.ipynb) | Deep head with 14 stylometric/static features | Paper-inspired feature engineering run |
| [src/task_A/Improved_models/codebert_deep_head_no_features_threshold_tuning.ipynb](src/task_A/Improved_models/codebert_deep_head_no_features_threshold_tuning.ipynb) | Deep head without extra features + threshold tuning | Decision-threshold calibration study |
| [src/task_A/Improved_models/codebert_deep_head_threshold_calibration.ipynb](src/task_A/Improved_models/codebert_deep_head_threshold_calibration.ipynb) | Deep head + focal/regularized training + calibrated threshold | High-performance calibrated Task A run |
| [src/task_A/Improved_models/graphcodebert_standard_overfitting.ipynb](src/task_A/Improved_models/graphcodebert_standard_overfitting.ipynb) | Standard GraphCodeBERT classifier | Explicit overfitting behavior check |
| [src/task_A/Improved_models/graphcodebert_focal_loss_multidropout.ipynb](src/task_A/Improved_models/graphcodebert_focal_loss_multidropout.ipynb) | GraphCodeBERT + focal loss + multi-sample dropout | Main robust GraphCodeBERT improved model |
| [src/task_A/Improved_models/graphcodebert_focal_loss_multidropout_outputs.ipynb](src/task_A/Improved_models/graphcodebert_focal_loss_multidropout_outputs.ipynb) | Same as above + saved outputs | Reproducibility/output snapshot |
| [src/task_A/Improved_models/graphcodebert_lolo_cpp.ipynb](src/task_A/Improved_models/graphcodebert_lolo_cpp.ipynb) | LOLO (leave-one-language-out) split with C++ holdout | Cross-language generalization stress test |
| [src/task_A/Improved_models/graphcodebert_meanpooling_lolo_javascript.ipynb](src/task_A/Improved_models/graphcodebert_meanpooling_lolo_javascript.ipynb) | Mean-pooling variant + LOLO with JavaScript holdout | Pooling/transfer robustness ablation |
| [src/task_A/Improved_models/train_2k_bilstm_deephead_different_backbones.ipynb](src/task_A/Improved_models/train_2k_bilstm_deephead_different_backbones.ipynb) | BiLSTM/deep-head across multiple backbones | Backbone comparison under low-data setup |
| [src/task_A/Improved_models/train_2k_smoothed_stylometric_feature_addition.ipynb](src/task_A/Improved_models/train_2k_smoothed_stylometric_feature_addition.ipynb) | Label smoothing + stylometric feature addition | Regularization + feature-addition ablation |
| [src/task_A/Improved_models/ensemble_run.ipynb](src/task_A/Improved_models/ensemble_run.ipynb) | Multi-model ensembling pipeline | Aggregate predictions for stronger final output |

### Task B

| Notebook | Model key features / innovations | Purpose |
|---|---|---|
| [src/task_B/baseline/Task-B-Baseline.ipynb](src/task_B/baseline/Task-B-Baseline.ipynb) | Baseline multiclass classifier | Task B baseline reference |
| [src/task_B/Improved_models/task_b_codebert_colab_baseline.ipynb](src/task_B/Improved_models/task_b_codebert_colab_baseline.ipynb) | CodeBERT pipeline tuned for Colab workflow | Portable baseline/improved starting point |
| [src/task_B/Improved_models/task_b_unixcoder_weighted_loss_hparam_search.ipynb](src/task_B/Improved_models/task_b_unixcoder_weighted_loss_hparam_search.ipynb) | UniXcoder + class-weighted loss + HP search | Main imbalance-aware UniXcoder pipeline |
| [src/task_B/Improved_models/task_b_phase1_loss_comparison.ipynb](src/task_B/Improved_models/task_b_phase1_loss_comparison.ipynb) | Side-by-side loss comparisons | Compare weighted/focal-style objectives |
| [src/task_B/Improved_models/task_b_phase1_codebert_unixcoder_weighted_loss_v1.ipynb](src/task_B/Improved_models/task_b_phase1_codebert_unixcoder_weighted_loss_v1.ipynb) | Early-phase CodeBERT/UniXcoder weighted-loss run | Phase-1 exploratory benchmark |
| [src/task_B/Improved_models/task_b_phase1_unixcoder_weighted_loss_overfitting_v1.ipynb](src/task_B/Improved_models/task_b_phase1_unixcoder_weighted_loss_overfitting_v1.ipynb) | UniXcoder weighted-loss variant | Overfitting diagnosis (version 1) |
| [src/task_B/Improved_models/task_b_phase1_unixcoder_weighted_loss_overfitting_v2.ipynb](src/task_B/Improved_models/task_b_phase1_unixcoder_weighted_loss_overfitting_v2.ipynb) | Same family with parameter/training changes | Overfitting diagnosis (version 2) |
| [src/task_B/Improved_models/GraphCodeBert/graphcodebert_task_b_baseline.ipynb](src/task_B/Improved_models/GraphCodeBert/graphcodebert_task_b_baseline.ipynb) | GraphCodeBERT baseline for Task B | Backbone-specific baseline |
| [src/task_B/Improved_models/GraphCodeBert/task_b_graphcodebert_focal_supcon_llrd_mixup.ipynb](src/task_B/Improved_models/GraphCodeBert/task_b_graphcodebert_focal_supcon_llrd_mixup.ipynb) | GraphCodeBERT + focal + SupCon + LLRD + mixup | Advanced imbalance/generalization pipeline |
| [src/task_B/Improved_models/GraphCodeBert/task_b_graphcodebert_focal_supcon_llrd_mixup_best.ipynb](src/task_B/Improved_models/GraphCodeBert/task_b_graphcodebert_focal_supcon_llrd_mixup_best.ipynb) | Best-performing tuned variant of above | Final high-performing Task B run |

### Task C

| Notebook | Model key features / innovations | Purpose |
|---|---|---|
| [src/task_C/baseline/task-C-Baseline.ipynb](src/task_C/baseline/task-C-Baseline.ipynb) | Baseline 4-class classifier | Task C baseline reference |
| [src/task_C/Improved_models/task-C_improved.ipynb](src/task_C/Improved_models/task-C_improved.ipynb) | Improved Task C training setup | Better separation of Human/Machine/Hybrid/Adversarial |

