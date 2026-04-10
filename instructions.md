# Running Instructions 

## How to Run (Best-Model Python Pipelines, Local)

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

## 3) Task A ensemble
Location: `src/task_A/Improved_models/ensemble`

What this does:
- Trains/uses 3 backbones (`codebert`, `graphcodebert`, `unixcoder`)
- Caches per-model probabilities
- Combines them with `soft_vote`, `weighted_avg`, or `rank_avg`

Important path note:
- This pipeline defaults to Kaggle-style paths from `config.py`.
- For local runs, first edit in `src/task_A/Improved_models/ensemble/config.py`:
  - `DATA_ROOT`, `TRAIN_PARQUET`, `TEST_PARQUET`
  - `OUTPUT_ROOT` (change from `/kaggle/working` to a local writable folder)

Run (full pipeline, train + ensemble):
1. `cd src/task_A/Improved_models/ensemble`
2. `python3 ensemble.py --strategy soft_vote`

Useful variants:
- Weighted averaging: `python3 ensemble.py --strategy weighted_avg --weights 0.4 0.35 0.25`
- Reuse cached probabilities only: `python3 ensemble.py --skip_training --strategy rank_avg`

Outputs:
- Submission CSV (default from config)
- Metadata JSON next to submission
- Cached probabilities in ensemble cache directory

## 4) Task B ensemble
Location: `src/task_B/Improved_models/ensemble_pipeline.py`

What this does:
- Loads multiple fine-tuned Task B checkpoints
- Generates per-model probability caches
- Runs ensemble (`soft_vote`, `weighted_avg`, `rank_avg`)
- Optional weight optimization and evaluation if labels are available

Run (single command full flow):
1. `cd src/task_B/Improved_models`
2. `python3 ensemble_pipeline.py full --model_paths <path_to_codebert_ckpt> <path_to_graphcodebert_ckpt> <path_to_unixcoder_ckpt> --parquet_path <task_b_parquet> --output_dir ./ensemble_cache_taskB --output_csv ./submission_b_ensemble.csv --strategy weighted_avg --task B`

Optional with validation labels (auto weight search + eval):
- add `--gold_csv <task_b_labels_csv>` to the `full` command

Other commands:
- cache predictions only: `python3 ensemble_pipeline.py predict ...`
- ensemble cached outputs only: `python3 ensemble_pipeline.py ensemble --prob_dir ./ensemble_cache_taskB --output_csv ./submission_b_ensemble.csv --strategy soft_vote --task B`
- optimize only: `python3 ensemble_pipeline.py optimize --prob_dir ./ensemble_cache_taskB --gold_csv <task_b_labels_csv>`
- evaluate only: `python3 ensemble_pipeline.py evaluate --submission_csv ./submission_b_ensemble.csv --gold_csv <task_b_labels_csv> --task B`
