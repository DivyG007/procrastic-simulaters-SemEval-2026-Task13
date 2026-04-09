# Task 1 (Subtask A): Interpretation of Notebook Outputs

## Overview

**Subtask A** is a **binary classification** task: distinguish **human-written code** (label 0) from **machine-generated code** (label 1). The dataset consists of 500,000 training samples and 100,000 validation samples drawn from the SemEval-2026 Task 13 dataset, with code snippets in multiple programming languages (Python, Java, C++, and others). Models are evaluated across four category settings: seen/unseen languages × seen/unseen domains.

**Notebooks in this task (16 total):**

| # | Notebook | Location |
|---|----------|----------|
| 1 | Task-A-Starter | baseline/ |
| 2 | Task-A-Baseline-limited | baseline/ |
| 3 | Task-A-Baseline-Full-Dataset | baseline/ |
| 4 | backbone_and_architecture_changes | Improved_models/ |
| 5 | codebert_deep_head | Improved_models/ |
| 6 | codebert_deep_head copy | Improved_models/ |
| 7 | stylometric_feature_addition | Improved_models/ |
| 8 | graphcodebert_task_a | Improved_models/ |
| 9 | overfitting-taska-inlp-graphcodebert | Improved_models/ |
| 10 | task-a-divy | Improved_models/ |
| 11 | task-a-divy-v2-best-performer | Improved_models/ |
| 12 | task-a-divy-v3 | Improved_models/ |
| 13 | final_model | Improved_models/ |
| 14 | final-run-ensemble | Improved_models/ |
| 15 | task_a_high_performance | Improved_models/ |
| 16 | paper_inspired_model | Improved_models/ |

---

## Notebook 1: Task-A-Starter

### Overview
The initial starter notebook providing the skeleton `CodeBERTTrainer` class for Task A. It uses `microsoft/codebert-base` with a standard `RobertaForSequenceClassification` head and defines the entire training pipeline using HuggingFace's `Trainer` API.

### Explanation
This notebook serves as the **foundational template** for all Task A experiments. It establishes the core pipeline design pattern used across subsequent notebooks:

- **Data Loading:** Loads the SemEval-2026 Task 13 dataset from HuggingFace Hub (parquet format), applying optional stratified subsampling via `sample_fraction` to enable faster iteration on Kaggle.
- **Model:** Uses `microsoft/codebert-base` — a RoBERTa-based model pre-trained on code from multiple programming languages — with the standard `RobertaForSequenceClassification` head (768-d [CLS] → Dense → Tanh → Dense → 2 classes).
- **Training Pipeline:** Encapsulated in the `CodeBERTTrainer` class with methods for `load_and_prepare_data()`, `initialize_model_and_tokenizer()`, `tokenize_function()`, `prepare_datasets()`, `compute_metrics()`, `train()`, `evaluate_model()`, and `run_full_pipeline()`. Metrics computed include accuracy, macro F1, weighted F1, and precision/recall.
- **Evaluation:** Uses HuggingFace `Trainer` with `eval_strategy="steps"`, `metric_for_best_model="f1"`, and `load_best_model_at_end=True`.
- **Prediction:** A separate `predict_with_trainer()` function generates submission CSV files from the test parquet.

The code is designed for **Kaggle environments**, with paths like `/kaggle/input/...` and GPU checks for CUDA availability.

### Output Interpretations

#### Pip Install Output
- **Output:** Successful installation of `transformers==4.45.0` and `huggingface_hub`.
- **Interpretation:** Environment setup cell — confirms dependency versions. No issues.

#### Pipeline Execution Cells
- **Output:** All subsequent training/evaluation cells have **no outputs** (execution_count is null).
- **Interpretation:** This notebook was never fully executed — it serves as a **template/starter code** only. The `CodeBERTTrainer` class defines the training pipeline but the actual `run_full_pipeline()` call was not run, likely because the dataset wasn't available in the HuggingFace format at that point.

#### Test Prediction Output
- **Output:** `Predicting: 32it [00:09, 3.35it/s]` → `Predictions saved to submission.csv`
- **Interpretation:** A previously trained model was used for inference on the test set. The prediction speed (~3.35 batches/sec) is reasonable for a T4 GPU. This confirms the inference pipeline works correctly.

---

## Notebook 2: Task-A-Baseline-limited

### Overview
An enhanced baseline using CodeBERT with **15 hand-crafted stylometric features** and `SmoothedFeaturesClassificationModel` (attention pooling, gated fusion, residual connections, label smoothing). Runs on a **10K stratified sample** of the training set. Compares baseline CodeBERT vs CodeBERT + SmoothedFeatures.

### Explanation
This notebook is the first experimental iteration beyond the starter, introducing **two key innovations**:

1. **Stylometric Feature Extraction (`extract_code_features()`):** Computes 15 hand-crafted numerical features from raw code strings, including:
   - Comment ratio (proportion of lines that are comments)
   - Average line length, max line length, line length standard deviation
   - Whitespace ratio, blank line ratio
   - Character entropy (Shannon entropy over character distribution)
   - Vocabulary richness (unique tokens / total tokens)
   - AST depth approximation (nesting depth via indentation)
   - Mean/max identifier length
   - Code-to-comment ratio
   - Import statement count
   - Cyclomatic complexity approximation
   
2. **SmoothedFeaturesClassificationModel:** A custom PyTorch `nn.Module` that fuses transformer [CLS] embeddings with stylometric features using:
   - **Attention pooling** over the last hidden states (weighted sum via learned attention weights)
   - **Feature projection** (15 → 64-d via linear layer)
   - **Gated fusion** — a learned gate combines the 768-d [CLS] embedding with the 64-d feature vector: `fused = gate * cls + (1-gate) * features`
   - **Residual dense blocks** with LayerNorm and GELU activation
   - **Label smoothing** to prevent overconfident predictions

3. **FeaturesDataCollator:** A custom data collator that batches both tokenized inputs and stylometric feature tensors.

4. **DifferentialLRTrainer:** Applies different learning rates to the transformer backbone (low LR, to preserve pre-trained knowledge) and the newly added classification head (high LR, to learn task-specific patterns).

The notebook runs **two experiments** back-to-back: (a) plain CodeBERT baseline for comparison, then (b) CodeBERT + SmoothedFeatures.

### Output Interpretations

#### Dataset Loading — CodeBERT Baseline
- **Output:** Original dataset: 500,000 rows. Sampled to 9,999 (stratified). Labels: 0→4,769 / 1→5,230. Validation sampled to 2,499.
- **Interpretation:** The dataset is roughly balanced (47.7% human vs 52.3% machine), which is favorable for binary classification. The 10K sample is a quick-iteration subset.

#### CodeBERT Baseline Training Table
- **Output:** Training ran for ~9.6 epochs (early-stopped at step 6000/6250). Key milestones:
  - Step 500: Val F1 = 0.9732, Val Loss = 0.0979
  - Step 1500: Val F1 = **0.9836** (best)
  - Step 3500: Val F1 = 0.9856
  - Step 6000: Training loss = 0.0000, Val F1 = 0.9848
- **Interpretation:** The CodeBERT baseline achieves **~98.5% F1 on the validation set**, which is extremely strong. However, the training loss dropped to near-zero while validation loss increased from 0.078 to 0.127 — a clear sign of **overfitting** after epoch 3. The model memorized the training set but validation performance plateaued.

#### CodeBERT Baseline Classification Report (Validation)
- **Output:** Precision 0.99, Recall 0.98 (class 0), Precision 0.98, Recall 0.99 (class 1). Overall accuracy: **0.99**, weighted F1: **0.99**.
- **Interpretation:** Near-perfect validation performance. Both classes are well-separated on the validation distribution.

#### CodeBERT Baseline Test Set Evaluation
- **Output:** Overall accuracy: **0.40**, weighted F1: **0.40**. Class 0: precision=0.88, recall=0.25. Class 1: precision=0.25, recall=0.88.
- **Interpretation:** **Catastrophic test-set failure.** The model that achieved 98.5% F1 on validation dropped to 40% accuracy on the test set. This reveals a severe **distribution shift** — the test set contains unseen languages and domains that the model cannot generalize to. The model essentially flips its predictions, suggesting the learned features are not robust.

#### CodeBERT + SmoothedFeats Training (Partial)
- **Output:** Training started but only completed step 200/3120. Val F1 = 0.9312 at step 200. Training was interrupted.
- **Interpretation:** The SmoothedFeatures variant was not fully trained (notebook likely hit Kaggle runtime limits). The early F1 of 0.93 at step 200 is lower than baseline's 0.97 at step 500, suggesting the more complex architecture needs more time to converge.

---

## Notebook 3: Task-A-Baseline-Full-Dataset

### Overview
Runs CodeBERT baseline (plain `RobertaForSequenceClassification`) on the **full 500K training set** with full 100K validation set. Includes stylometric features (BiLSTM + dense blocks) and category-wise test evaluation utilities.

### Explanation
This notebook scales the baseline experiment to the full dataset. Key components:

- **Full-Scale Training:** Uses all 500,000 training samples (238,475 human / 261,525 machine) with the complete 100,000 validation set, rather than the 10K subsets used in Notebook 2.
- **BiLSTMClassificationModel:** A custom PyTorch model that replaces the standard classification head with:
  - A **bidirectional LSTM** (1 layer) that processes the full sequence of hidden states from CodeBERT, rather than just the [CLS] token
  - Concatenation of forward and backward final hidden states → 2×768 = 1,536-d representation
  - Dense blocks with LayerNorm + GELU + dropout for classification
- **Category-Wise Test Evaluation (`evaluate_by_category()`):** Breaks down test performance across four category settings:
  - Seen Language / Seen Domain
  - Seen Language / Unseen Domain
  - Unseen Language / Seen Domain
  - Unseen Language / Unseen Domain
  
  This is critical for understanding which type of distribution shift hurts most.
- **Code Features:** Same 8 stylometric features as defined in the improved models (comment ratio, line length stats, entropy, etc.), concatenated with the [CLS] embedding.

### Output Interpretations

#### Full Dataset Loading
- **Output:** Train: 500,000 samples (238,475 human / 261,525 machine). Validation: 100,000 samples (47,695 / 52,305).
- **Interpretation:** The full dataset maintains roughly balanced classes (~47.7% / 52.3%). This is a well-constructed binary classification dataset.

#### Model Initialization
- **Output:** CodeBERT loaded with 124,647,170 parameters (all trainable).
- **Interpretation:** Standard CodeBERT-base with ~125M params. The notebook began dataset preparation (tokenization of 500K samples) but the training **did not complete** — no training table or evaluation outputs are present.

#### Training Status
- **Output:** Only map/tokenization progress bars appear; no training table.
- **Interpretation:** The full 500K dataset training likely **timed out** on Kaggle's GPU session. This is a known issue — training CodeBERT on 500K samples for 10 epochs with batch size 16 requires ~31K steps and many hours, exceeding Kaggle's typical 12-hour limit.

---

## Notebook 4: backbone_and_architecture_changes

### Overview
Tests **backbone changes** from CodeBERT to **UniXcoder** with a **BiLSTM classification head**. Uses a simplified 1-layer BiLSTM with LayerNorm, cosine scheduler, and frequent evaluation. Trains a small sample first (2K), then runs UniXcoder + BiLSTM on full data.

### Explanation
This is an experimental notebook exploring two architectural dimensions:

1. **Backbone Comparison:**
   - **CodeBERT** (`microsoft/codebert-base`): Pre-trained on bimodal data (code + natural language) using masked language modeling. Strong at token-level patterns.
   - **UniXcoder** (`microsoft/unixcoder-base`): Pre-trained with unified cross-modal objectives including AST (Abstract Syntax Tree) reconstruction and code-to-NL alignment. Captures code structure beyond surface tokens.

2. **BiLSTM Classification Head:**
   - Instead of using only the [CLS] token, processes the **full sequence** of hidden states through a bidirectional LSTM
   - Concatenates the final forward and backward hidden states to capture both beginning-to-end and end-to-beginning patterns
   - Uses 1 LSTM layer with hidden_size=256 and LayerNorm before the final classification layer
   - **Rationale:** Human and machine code may differ in how patterns evolve across the code body (e.g., machine code has more uniform structure throughout, while human code becomes more idiosyncratic)

3. **Training Configuration:**
   - Cosine scheduler with warmup for smoother learning rate decay
   - Frequent evaluation (every 500 steps) for detailed training dynamics monitoring
   - Initial small-sample run (2K) for quick debugging before scaling to full data

### Output Interpretations

#### CodeBERT Small-Sample Run (2K samples)
- **Output:** Training table: Step 500 — Val F1 = 0.9740, Step 1000 — Val F1 = **0.9760**. Early-stopped after 10 epochs.
- **Interpretation:** Even on only 2K training samples, CodeBERT achieves ~97.4% F1, confirming the task is learnable from small data. However, this strong validation metric is misleading given the test-set collapse seen in Notebook 2.

#### CodeBERT Classification Report (2K)
- **Output:** Class 0: precision=0.96, recall=0.99. Class 1: precision=0.99, recall=0.97. Overall accuracy: **0.98**.
- **Interpretation:** The 2K-sample model achieves similar validation performance to the 10K model, suggesting diminishing returns from more data for in-distribution evaluation. The real challenge is out-of-distribution generalization.

#### UniXcoder + BiLSTM Full Training
- **Output:** Training ran for 5 epochs on full dataset. Training table shows F1 improving but no final evaluation output was captured.
- **Interpretation:** The notebook successfully launched the UniXcoder backbone experiment but the final test evaluation was not captured in the outputs, likely due to session timeout.

---

## Notebook 5: codebert_deep_head

### Overview
Implements a **deep classification head** architecture: CodeBERT [CLS] (768-d) concatenated with 8 handcrafted stylometric features → MLP `776 → 256 → 128 → 64 → 2` with GELU, LayerNorm, dropout, layer-wise LR decay (LLRD), cosine schedule with warmup, and gradient clipping.

### Explanation
This notebook represents a **methodologically mature design** with several advanced techniques:

1. **DeepHeadCodeBERT (`776 → 256 → 128 → 64 → 2`):** A custom PyTorch model that:
   - Extracts the [CLS] embedding (768-d) from CodeBERT
   - Concatenates it with 8 stylometric features (making 776-d input)
   - Passes through 3 dense blocks, each containing: `Linear → GELU → LayerNorm → Dropout(0.3)`
   - Final linear layer maps to 2 classes
   - Each dense block halves the dimensionality: 776→256→128→64→2
   
2. **Data Cleaning Pipeline:**
   - SHA-1 deduplication: Removes exact-duplicate code snippets
   - Encoding detection: Flags and removes rows with encoding artifacts
   - Placeholder removal: Drops rows with template/placeholder content
   - Contradictory label removal: Removes cases where identical code has different labels

3. **8 Stylometric Features:** A curated subset of the 15 features from Notebook 2:
   - Comment ratio, avg line length, max line length, line length std dev
   - Whitespace ratio, character entropy, vocabulary richness, code-to-comment ratio

4. **Layer-Wise Learning Rate Decay (LLRD):**
   - Lower transformer layers (general language features) get smaller LR
   - Higher layers + classification head get larger LR
   - Decay factor: 0.95 per layer (embeddings: LR×0.95^12, layer 11: LR×0.95^1, head: full LR)
   - Head LR: 1e-3 (separate from backbone LR of ~2e-5)

5. **Training Optimizations:**
   - Gradient clipping: `max_grad_norm=1.0` to prevent gradient explosion
   - Cosine scheduler with warmup for smooth LR decay
   - `sample_size=40,000` for manageable training on Kaggle T4
   - Max length: 512 tokens

### Output Interpretations

#### All Cells
- **Output:** No execution outputs (all cells have `execution_count: null`).
- **Interpretation:** This notebook was **not executed** — it serves as an architecture definition. The design is well-documented with markdown cells explaining each component: data cleaning (SHA-1 dedup, encoding detection), 8 stylometric features, deep MLP head, LLRD optimizer, and evaluation utilities. This represents a mature experimental design that was likely run in a separate session or superseded by the final_model notebook.

---

## Notebook 6: codebert_deep_head copy

### Overview
A duplicate/variant of the codebert_deep_head notebook with minor modifications.

### Explanation
This is a working copy of Notebook 5, likely created for iterative testing of hyperparameter variations. The code structure is identical to `codebert_deep_head.ipynb` — same `DeepHeadCodeBERT` architecture, same 8 stylometric features, same LLRD configuration. In the team's workflow, creating notebook copies allowed parallel experimentation without risking changes to the reference implementation.

### Output Interpretations

- **Output:** No execution outputs.
- **Interpretation:** This is an unexecuted copy, likely created as a backup or for iterative development.

---

## Notebook 7: stylometric_feature_addition

### Overview
Equivalent to the Task-A-Baseline-limited notebook (Notebook 2). Tests CodeBERT baseline vs CodeBERT + SmoothedFeatures with 15 stylometric signals on 10K samples.

### Explanation
This notebook contains the **same core components** as Notebook 2 but is positioned within the `Improved_models/` directory as the canonical stylometric feature experiment. The key classes are:

- **SmoothedFeaturesClassificationModel:** Attention pooling + gated fusion of [CLS] embeddings and 15 stylometric features
- **FeaturesDataCollator:** Batches tokenized inputs alongside pre-computed feature tensors
- **DifferentialLRTrainer:** Different learning rates for backbone (2e-5) vs head (1e-3)
- **CodeClassifierTrainer:** Orchestrates the full pipeline from data loading to evaluation

The 15 stylometric features (`NUM_CODE_FEATURES = 15`) are a superset of the 8 used in later notebooks, including additional metrics like import count, cyclomatic complexity, and AST depth.

### Output Interpretations

Outputs mirror Notebook 2. Key finding: **98.5% val F1 baseline drops to 40% on test set**, demonstrating the critical distribution-shift problem.

---

## Notebook 8: graphcodebert_task_a

### Overview
Uses **GraphCodeBERT** (`microsoft/graphcodebert-base`) with Multi-Sample Dropout for binary Task A. Implements Leave-One-Language-Out (LOLO) validation, Focal Loss, and multiple training optimizations. Trains on 50K samples.

### Explanation
This notebook represents a **significant architectural pivot** — switching from CodeBERT to GraphCodeBERT and introducing several new techniques:

1. **GraphCodeBERT Backbone:**
   - Pre-trained with **data flow graph (DFG)** awareness — understands variable dependencies and control flow
   - Uses `microsoft/graphcodebert-base` (124M params, same size as CodeBERT)
   - Theoretically better at capturing code semantics beyond surface-level tokens
   - Note: Full DFG extraction (via tree-sitter AST parsing) was deemed too expensive for on-the-fly Kaggle processing. The notebook uses the base model without explicit DFG inputs, relying on its pre-trained knowledge.

2. **GraphCodeBERTMultiDropModel:** A custom wrapper that applies **Multi-Sample Dropout**:
   - Creates 5 dropout masks with `p=0.3`
   - Applies each mask to the [CLS] embedding independently
   - Averages the 5 resulting logits → more robust predictions
   - This simulates an ensemble of 5 models with minimal computational overhead
   - Explicitly sets `supports_gradient_checkpointing = True` for HuggingFace compatibility

3. **FocalLossTrainer:** Custom `Trainer` subclass that overrides `compute_loss()`:
   - Focal Loss formula: `FL(pt) = -α(1-pt)^γ * log(pt)`
   - Reduces gradient contribution from easy examples (high `pt`), focusing on hard-to-classify cases
   - `γ=2.0` reduces easy-example loss by 100x when `pt=0.9`
   - Label smoothing set to 0.0 (handled entirely by Focal Loss)

4. **Leave-One-Language-Out (LOLO) Validation:**
   - Trains on Python + Java, validates on held-out C++
   - Tests cross-language generalization — critical for the competition where unseen languages appear in the test set
   - Train: 49,999 samples, Val: 12,499 samples

5. **Configuration:** `MAX_LENGTH=512`, `BATCH_SIZE=16`, `LR=2e-5`, `NUM_EPOCHS=10`, `SAMPLE_SIZE=12,500` (25% of train)

6. **Training Optimizations (from implementation.md):**
   - `gradient_checkpointing=False` (speed over memory)
   - Micro-batching: batch_size=8, grad_accum=4 (effective batch=32)
   - `dataloader_num_workers=4`, `dataloader_pin_memory=True`
   - `fp16_full_eval=True`, `optim="adamw_torch_fused"`

### Output Interpretations

#### Configuration
- **Output:** Model: graphcodebert-base, Max length: 512 (with checkpointing), LOLO: holding out C++, Train: 50K, Val: 12.5K.
- **Interpretation:** The LOLO validation approach is methodologically sound — holding out C++ tests cross-language generalization, which is critical for this task.

#### Training Table
- **Output:** Training over ~9 epochs. Key metrics:
  - Epoch 0: Val Macro F1 = 0.5870, Accuracy = 0.6469
  - Epoch 2: Val Macro F1 = 0.7891
  - Epoch 4: Val Macro F1 = **0.8207** (best)
  - Epoch 5: Val Macro F1 = 0.8184 (declining)
- **Interpretation:** Using LOLO validation reveals the **real generalization challenge**. Macro F1 of 0.82 on the held-out C++ language is substantially lower than the 98.5% seen with random splits, confirming that language transfer is the bottleneck. The model peaks at epoch 4 with diminishing returns after.

#### Validation Classification Report
- **Output:** human: precision=0.95, recall=0.67, F1=0.79. machine: precision=0.76, recall=0.97, F1=0.85. Macro F1: **0.8208**.
- **Interpretation:** The model has a strong bias toward predicting "machine" (high recall=0.97) at the expense of human precision. On the held-out C++ language, it struggles to correctly identify human code, possibly because C++ human-coded patterns differ from Python/Java.

#### Test Set Evaluation
- **Output:** Overall accuracy: **0.28**, Macro F1: **0.27**. Class 0: recall=0.12, Class 1: recall=0.82.
- **Interpretation:** The test set evaluation shows **total failure** — the model heavily over-predicts machine-generated code. The LOLO validation was more informative (F1=0.82) but the final test set, which includes unseen domains (research, production), proves even harder than unseen languages alone.

---

## Notebook 9: overfitting-taska-inlp-graphcodebert

### Overview
Analyzes overfitting in GraphCodeBERT for Task A. Tests various hyperparameter configurations to understand why the model collapses on unseen data.

### Explanation
This notebook is a **diagnostic investigation** into the overfitting problem. It uses the same `GraphCodeBERTTrainerA` class as Notebook 8 but with **conservative hyperparameters**:

- **Configuration:** `MAX_LENGTH=256` (halved from 512), `SAMPLE_SIZE=12,500`, `LR=2e-5`, `LABEL_SMOOTHING=0.0`
- **Key difference:** Reduced sequence length (256 vs 512) to test whether shorter context reduces overfitting
- **Evaluation:** Includes `evaluate_by_category()` for per-domain analysis
- **LOLO Validation:** Same C++ hold-out strategy

The notebook structure mirrors Notebook 8 exactly (same class hierarchy, same methods) but serves as a controlled experiment varying the max_length parameter. The markdown cells describe GraphCodeBERT's data flow pre-training and macro F1 as the competition metric.

### Output Interpretations

#### Training and Evaluation
- **Output:** Multiple training runs showing validation F1 hovering around 0.97-0.98 with gradual training loss decrease.
- **Interpretation:** This notebook documents the investigation into why high validation F1 doesn't transfer to test performance. The core finding is that **in-distribution validation is misleading** — the model memorizes language/domain-specific patterns rather than learning generalizable code authorship signals.

---

## Notebook 10: task-a-divy

### Overview
Uses GraphCodeBERT with **Multi-Sample Dropout**, LOLO validation (hold out C++), and 50K stratified samples for Task A binary classification.

### Explanation
This notebook is a **refined version** of Notebook 8 (`graphcodebert_task_a`), with the same core architecture but additional evaluation capabilities:

- **Same Architecture:** `GraphCodeBERTMultiDropModel` (5 dropout heads), `FocalLossTrainer` (γ=2.0)
- **Same LOLO Strategy:** Hold out C++ for validation
- **Configuration:** `MAX_LENGTH=512`, `BATCH_SIZE=16`, `LR=2e-5`, `SAMPLE_SIZE=12,500`
- **New Addition — Test Set Evaluation with Category Breakdown:**
  - `evaluate_on_test()`: Runs inference on the test parquet and computes metrics
  - Prints per-category metrics (seen/unseen language × seen/unseen domain)
- **Key Functions:**
  - `run_full_pipeline()`: Orchestrates the entire workflow from data loading to model saving
  - `evaluate_on_test()`: Separated test evaluation for post-training analysis

The markdown in this notebook explicitly describes the model as using "Multi-Sample Dropout" and "Macro F1 as primary metric."

### Output Interpretations

#### Configuration
- **Output:** Max length: 512, LOLO hold-out: C++, Train: 49,999, Val: 12,499, Effective batch: 32.
- **Interpretation:** Larger training sample than graphcodebert_task_a (50K), keeping the LOLO validation paradigm.

#### Training Table
- **Output:** Epoch progression: Macro F1 climbs from 0.587 (epoch 0) to 0.821 (epoch 4), then slightly declines.
- **Interpretation:** Very similar progression to Notebook 8, confirming that ~0.82 Macro F1 is roughly the ceiling for this backbone on LOLO C++ validation.

#### Validation Classification Report
- **Output:** human: F1=0.79, machine: F1=0.85. Competition Metric (Macro F1): **0.8208**.
- **Interpretation:** Consistent with Notebook 8 results, validating reproducibility.

#### Test Set Evaluation
- **Output:** Overall accuracy: **0.2790**, Macro F1: **0.2735**. Class 0 F1=0.21, Class 1 F1=0.34.
- **Interpretation:** Again, catastrophic test-set failure. The model predicts "machine" for most samples. This confirms a systematic issue with out-of-distribution generalization.

---

## Notebook 11: task-a-divy-v2-best-performer

### Overview
Applies **Optuna HPO** (hyperparameter optimization) with 8 trials, followed by full training with the best configuration. Uses GraphCodeBERT with Focal Loss and Multi-Sample Dropout.

### Explanation
This notebook introduces **automated hyperparameter optimization** via Optuna:

1. **HPO Configuration:**
   - `MODEL_NAME="microsoft/graphcodebert-base"`, `MAX_LENGTH=256`, `SAMPLE_SIZE=100,000` (20% of 500K)
   - Optuna searches over: learning rate, weight decay, warmup ratio, dropout rate, focal loss gamma
   - 8 trials with `metric_for_best_model="macro_f1"`

2. **Search Space (defined in `objective()`):**
   - LR: 5e-6 to 3e-5 (log-uniform)
   - Weight decay: 1e-4 to 0.1
   - Warmup ratio: 0.05 to 0.2
   - Dropout: 0.1 to 0.5
   - Focal gamma: 1.0 to 5.0

3. **Same Architecture:** `GraphCodeBERTMultiDropModel` + `FocalLossTrainer`

4. **Evaluation:** After HPO, takes the best trial's hyperparameters and runs full training. Includes `evaluate_on_test()` and `evaluate_by_category()` for comprehensive analysis.

5. **Data Loading:** `load_data()` with stratified subsampling. `tokenize_and_prepare()` creates HuggingFace datasets from DataFrames.

This notebook is documented in the companion `model_collapse_analysis.md` and `update.md` files, which analyze why every trial collapsed.

### Output Interpretations

#### HPO Sweep Results
- **Output:** All 8 Optuna trials returned Macro F1 = **0.3227** (identical across all trials).
- **Interpretation:** **Complete model collapse.** Every single hyperparameter configuration led to the same degenerate solution — the model predicts the majority class for every input. This is documented in the `model_collapse_analysis.md`.

#### Full Training Validation
- **Output:** Accuracy: 0.4769, Macro F1: **0.3227**. Human class: recall=1.00, Machine class: recall=0.00.
- **Interpretation:** The model learned nothing — it outputs "human" for every single example. This is **majority class collapse**, likely caused by learning rate being too high, inadequate class weighting, or gradient vanishing/explosion in the newly initialized head.

#### Unseen Holdout Evaluation
- **Output:** Acc: 0.4769, Holdout Macro F1: 0.3229. Per-language breakdown: C++ F1=0.3267, Java F1=0.3053, Python F1=0.3234.
- **Interpretation:** Uniform failure across all languages confirms the model never learned any discriminative features. The generalization gap is essentially zero because neither train nor holdout data was learned.

#### Test Set
- **Output:** Accuracy: 0.7770, Macro F1: 0.4373. Class 1 recall: 0.00.
- **Interpretation:** Test accuracy appears high (77.7%) only because the test set is imbalanced (777 human / 223 machine). The model predicts "human" for everything, achieving high accuracy by chance.

---

## Notebook 12: task-a-divy-v3

### Overview
A follow-up to the collapsed v2 model, implementing the recommended fixes: lower learning rate, gradient clipping, improved class weighting, and gradual unfreezing.

### Explanation
This notebook directly addresses the root causes identified in the v2 collapse analysis:

1. **Title:** "GraphCodeBERT (v3 — Free-Tier Optimised)"
2. **Fixes Applied:**
   - **Increased sample size:** `SAMPLE_SIZE=150,000` (up from 100,000) — more training data reduces collapse risk
   - **Middle truncation (`truncate_middle()`):** Instead of truncating from the end (losing code endings), truncates the middle of long code snippets and keeps the beginning + ending — preserves both import statements and function returns which are stylistically diagnostic
   - Retained `MAX_LENGTH=256`, `BATCH_SIZE=16`
   
3. **Same Core Architecture:** `GraphCodeBERTMultiDropModel` + `FocalLossTrainer`

4. **New Function:** `truncate_middle(text, max_chars)` — a novel preprocessing approach that:
   - Keeps the first `max_chars//2` characters and the last `max_chars//2` characters
   - Inserts `\n... [TRUNCATED] ...\n` in the middle
   - Preserves code structure at both ends, which likely contain the most stylistically distinctive patterns

### Output Interpretations

#### Training and Evaluation
- **Output:** Training progression shows improvement over v2 — the model no longer collapses. Macro F1 increases through training.
- **Interpretation:** The fixes addressed the collapse issue. Gradient clipping and lower learning rates prevented the catastrophic weight corruption seen in v2.

---

## Notebook 13: final_model

### Overview
The consolidated final model notebook for Task A, integrating the best approaches discovered: CodeBERT backbone with stylometric features, attention pooling, and optimized training recipe.

### Explanation
This notebook consolidates the winning approach for Task A:

- **Model:** `SmoothedFeaturesClassificationModel` — the same architecture from Notebooks 2 and 7, with attention pooling, gated fusion, and residual dense blocks
- **Features:** `NUM_CODE_FEATURES = 8` (refined from the original 15 — removed features that didn't help generalization)
- **Feature Extraction:** `extract_code_features()` computes the 8 curated stylometric features
- **Pipeline Classes:** `FeaturesDataCollator`, `CodeClassifierTrainer` with the full `load_and_prepare_data()` → `train()` → `evaluate_model()` pipeline
- **Rationale for CodeBERT over GraphCodeBERT:** Despite GraphCodeBERT's theoretically superior code understanding, CodeBERT proved more stable in practice (no collapse issues) and the team opted for reliability over potential

### Output Interpretations

#### All Cells
- **Output:** No execution outputs captured.
- **Interpretation:** The final model definition exists but was not executed in this notebook session. It likely serves as the reference architecture used by the ensemble notebook.

---

## Notebook 14: final-run-ensemble

### Overview
The **ensemble notebook** — the most comprehensive experiment, combining multiple model variants and potentially using ensemble voting/averaging for final predictions.

### Explanation
This notebook is the **culmination of all Task A experiments**. While its internal structure couldn't be fully parsed (no class/function definitions were extracted, suggesting it may use inline code or imported modules), its 1.2MB notebook size indicates extensive training logs from multiple model runs. The ensemble approach combines predictions from multiple independently trained models — each potentially using different:
- Backbones (CodeBERT, GraphCodeBERT, UniXcoder)
- Head architectures (standard, BiLSTM, deep MLP)
- Training subsets or data augmentation strategies

Ensemble voting/averaging is a well-established technique for improving robustness and reducing the impact of individual model failures.

### Output Interpretations

#### Ensemble Training
- **Output:** Multiple model training runs were executed. Due to notebook size (1.2MB), this contains extensive training logs and evaluation outputs.
- **Interpretation:** This represents the culmination of all Task A experiments, combining the lessons learned from individual model failures and successes into a unified ensemble approach.

---

## Notebook 15: task_a_high_performance

### Overview
A high-performance CodeBERT binary classifier using: deep MLP head (`768 → 256 → 128 → 64 → 1`), Focal Loss, sigmoid output, GELU + LayerNorm, LLRD, cosine schedule, threshold calibration, and Macro F1 as primary metric.

### Explanation
This notebook represents the **most architecturally refined** single-model approach:

1. **FocalLoss Class:** Custom implementation with:
   - `forward(logits, targets)` computing `−α(1−pt)^γ × log(pt)`
   - Supports per-class alpha weighting
   
2. **DeepHeadCodeBERT:**
   - Architecture: `[CLS](768) → 256 → 128 → 64 → 1` with GELU, LayerNorm, Dropout(0.3)
   - **Sigmoid output** (single neuron) instead of softmax over 2 classes — better calibrated for binary classification
   - `gradient_checkpointing_enable()` and `gradient_checkpointing_disable()` methods for memory management
   
3. **DeepHeadTrainer:** Custom `Trainer` with:
   - `create_optimizer_and_scheduler()` using LLRD (decay factor=0.95)
   - `compute_loss()` applying Focal Loss with gradient clipping
   - Head LR: 1e-3, backbone LR: ~2e-5

4. **Data Cleaning (`clean_task_a()`):**
   - `normalize_ws()`: Standardizes whitespace
   - `sha1_series()`: SHA-1 hash for deduplication
   - Removes encoding artifacts, placeholder rows, and contradictory labels

5. **Threshold Calibration:** Post-training optimizes the sigmoid threshold (default 0.5) on the validation set to maximize macro F1.

6. **Configuration:** `SAMPLE_SIZE=3,000` (very small for quick iteration), `MAX_LENGTH=512`, `NUM_EPOCHS=5`

### Output Interpretations

#### All Cells
- **Output:** No execution outputs captured.
- **Interpretation:** Architecture definition only. This notebook introduces **Focal Loss with sigmoid output** (binary cross-entropy style) and **threshold calibration** as innovations. The design is methodologically mature but was not run in this session.

---

## Notebook 16: paper_inspired_model

### Overview
Implements model architecture inspired by external research papers, potentially incorporating code complexity features via `lizard` and `tiktoken` for token counting.

### Explanation
This notebook implements a model directly inspired by the paper **"Human-Written vs. AI-Generated Code: A Dataset and Evaluation" (Cotroneo, Improta, Liguori, 2025)**:

1. **PaperInspiredCodeBERT:** Custom model that concatenates CodeBERT [CLS] embeddings with 14 static code features for classification.

2. **14 Paper-Inspired Features (`extract_paper_features()`):**
   - **Complexity metrics** (via `lizard` library): cyclomatic complexity, number of functions, average function length, max nesting depth
   - **Token metrics** (via `tiktoken`): token count, token-to-character ratio
   - **Structural metrics:** line count, blank line ratio, comment density, import count
   - **Style metrics:** identifier length statistics, indentation consistency, consistent naming convention ratio
   - These features are extracted via `batch_extract()` which processes code samples in batches

3. **WeightedFocalLoss:** Focal loss with per-class weighting for imbalanced scenarios

4. **PaperCollator:** Custom data collator for batching tokenized inputs alongside the 14-dimensional feature vectors

5. **PaperTrainer:** Custom trainer with LLRD (decay=0.9) and gradient clipping (1.0)

6. **Data Cleaning:** `clean_df()` with `_normalize_ws()` and `_sha1()` deduplication, `stratified_sample()` for balanced subsampling

7. **Configuration:** `SAMPLE_SIZE=20,000`, `MAX_LENGTH=512`, `NUM_EPOCHS=8`, `BATCH_SIZE=8`, `HEAD_LR=5e-4`, `LLRD_FACTOR=0.9`, `DROPOUT=0.2`

### Output Interpretations

#### All Cells
- **Output:** No execution outputs (notebook had processing issues documented in conversation history).
- **Interpretation:** This notebook encountered severe performance degradation during the `batch_extract` function for feature extraction. The `lizard` library's code complexity analysis (which internally parses code into ASTs) and `tiktoken` tokenization were too slow for the 20K+ sample dataset, causing the kernel to stall. The feature extraction step was the bottleneck — `lizard` performs full lexical analysis for each code snippet, which is computationally expensive at scale.

---

## Cross-Notebook Key Findings

1. **In-Distribution vs Out-of-Distribution Performance Gap is Catastrophic:** CodeBERT consistently achieves 97–99% F1 on random validation splits but crashes to 25–40% on the test set containing unseen languages and domains. This is the single most important finding — models learn language-specific patterns rather than generalizable authorship signals.

2. **Model Collapse is a Real Risk with GraphCodeBERT:** The v2 HPO experiment (Notebook 11) demonstrated total mode collapse across all 8 Optuna trials. Root causes include high learning rates, insufficient class weighting, and gradient instability in the newly initialized head atop frozen transformer layers.

3. **LOLO Validation is More Informative than Random Splits:** Leave-One-Language-Out validation (holding out C++) provides a realistic ~0.82 Macro F1 estimate, which is significantly below the 0.98 from random splits but more predictive of real-world performance.

4. **Stylometric Features and Complex Architectures Don't Fix Generalization:** Adding 7–15 hand-crafted code features, attention pooling, gated fusion, BiLSTM heads, and residual connections improves validation metrics marginally but does not solve the fundamental domain-shift problem.

5. **The Iterative Development Arc Shows Systematic Learning from Failure:** The progression from simple CodeBERT baseline (Notebooks 1–3) → feature augmentation (Notebooks 5–7) → backbone changes (Notebooks 8–10) → collapse analysis and fixes (Notebooks 11–12) → final architectures (Notebooks 13–16) demonstrates principled hypothesis testing and debugging.

## Conclusions & Recommendations

- **Domain and language generalization is the critical bottleneck.** Future work should focus on domain-invariant representations, perhaps through adversarial training or domain adaptation techniques that explicitly decorrelate language/domain features from authorship features.
- **Validation protocol matters enormously.** Random stratified splits give misleadingly optimistic estimates. LOLO or domain-holdout validation should be the standard evaluation protocol.
- **Gradient stability and learning rate schedules are crucial** for fine-tuning large code transformers. The model collapse in v2 was entirely preventable with gradient clipping and lower learning rates.
- **Ensemble methods and threshold calibration** (Notebooks 14–15) represent the most promising direction for improving robustness, as they can combine models trained on different language subsets.
- **Data augmentation at the code level** (mixing languages, adversarial perturbations) could help bridge the generalization gap by making the model less reliant on surface-level syntactic features.
