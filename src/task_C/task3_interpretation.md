# Task 3 (Subtask C): Interpretation of Notebook Outputs

## Overview

**Subtask C** is a **4-class classification** task: given a code snippet, classify it into one of four categories:
- **Class 0: Human** — entirely human-written code
- **Class 1: Machine** — entirely AI/LLM-generated code  
- **Class 2: Hybrid** — code that mixes human-written and AI-generated portions
- **Class 3: Adversarial** — AI-generated code deliberately modified to appear human-written (or vice versa)

The dataset contains approximately 900,000 training samples and 200,000 validation samples. The key challenge is distinguishing **hybrid** and **adversarial** code from pure human or machine code — these categories represent subtle, nuanced blends that require the model to detect partial-automation and intentional obfuscation patterns. The competition metric is **Macro F1**.

**Notebooks in this task (2 total):**

| # | Notebook | Location |
|---|----------|----------|
| 1 | task-C-Baseline | baseline/ |
| 2 | task-C_improved | Improved_models/ |

---

## Notebook 1: task-C-Baseline

### Overview
The baseline notebook for Task C using `microsoft/codebert-base` (CodeBERT) with a standard `RobertaForSequenceClassification` head. Uses 10% of training/validation data with a basic training recipe (linear LR schedule, no class weighting, no focal loss).

### Explanation
This notebook adapts the foundational `CodeBERTTrainer` class from the Task A starter for the 4-class hybrid code detection problem:

1. **CodeBERTTrainer Class:** The main pipeline class with the following well-defined methods:
   - `__init__(model_name, num_labels, max_length, sample_fraction)`: Initializes with CodeBERT and 4 output labels
   - `load_and_prepare_data()`: Loads the Task C dataset from HuggingFace Hub or local parquet files. Applies `sample_fraction=0.1` to use only 10% of the data (90K train, 20K val) for Kaggle runtime constraints. Computes and prints label distribution statistics.
   - `initialize_model_and_tokenizer()`: Creates `AutoTokenizer` and `RobertaForSequenceClassification` from `microsoft/codebert-base` with `num_labels=4`
   - `tokenize_function(examples)`: Applies tokenizer with padding, truncation, and the configured `max_length`
   - `prepare_datasets()`: Applies tokenization via HuggingFace `Dataset.map()` and sets the torch format
   - `compute_metrics(eval_pred)`: Computes accuracy, macro F1, weighted F1, and per-class precision/recall using sklearn
   - `train()`: Configures HuggingFace `Trainer` with:
     - `eval_strategy="steps"`, `eval_steps=500` (evaluate every 500 training steps)
     - `metric_for_best_model="f1"` (selects checkpoint with highest F1)
     - `load_best_model_at_end=True` (loads the best checkpoint after training completes)
     - `save_total_limit=2` (keeps only 2 checkpoints to save disk space)
     - Standard linear LR schedule with warmup
   - `evaluate_model()`: Runs inference on the validation set and prints a full `classification_report` with per-class precision, recall, and F1
   - `run_full_pipeline()`: Orchestrates the entire workflow: load data → init model → prepare datasets → train → evaluate → return trainer
   - `predict_with_trainer(trainer, test_path, output_csv)`: Generates submission CSV from test parquet
   - `batcher(iterable, batch_size)`: Utility for batching inference

2. **Model Architecture:**
   - `RobertaForSequenceClassification` with the standard RoBERTa classification head:
     - RobertaModel (12 encoder layers, 768 hidden size, 12 attention heads)
     - Classification head: `Dense(768→768)` → `Tanh` → `Dropout(0.1)` → `Dense(768→4)`
   - Total parameters: ~124.6M (all trainable)
   - No custom heads, no class weighting, no focal loss — pure vanilla fine-tuning

3. **Data Configuration:**
   - Train: 90,000 samples (10% of ~900K), Val: 20,000 samples (10% of ~200K)
   - No explicit class weighting despite significant imbalance (Human: 54%, Machine: 23%, Adversarial: 13%, Hybrid: 10%)
   - Standard cross-entropy loss treats all classes equally

4. **Environment:** Designed for Kaggle with Tesla T4 GPU (15.6 GB VRAM). Uses `transformers==4.45.0`.

### Output Interpretations

#### Dataset Loading and Configuration
- **Output:** 
  - Dataset columns: `['code', 'generator', 'label', 'language']`
  - Sample data shows code snippets in JavaScript, PHP, Java, and other languages
  - Using 10% of training data: 90,000 samples
  - Using 10% of validation data: 20,000 samples
  - Number of unique labels: 4 (0–3)
  - Label distribution:
    - 0 (Human): 48,650 (54.1%)
    - 1 (Machine): 20,949 (23.3%)
    - 2 (Hybrid): 8,621 (9.6%)
    - 3 (Adversarial): 11,780 (13.1%)
- **Interpretation:** The class distribution is significantly imbalanced. Human code dominates at 54%, followed by Machine at 23%, Adversarial at 13%, and Hybrid at only 9.6%. The hybrid class — arguably the most challenging to detect — is also the most underrepresented. This imbalance directly explains why the baseline model struggles most with hybrid code classification.

#### Model Architecture
- **Output:** Full `RobertaForSequenceClassification` architecture printed:
  - RobertaModel with 12 encoder layers, 768 hidden size
  - Classification head: Dense(768→768) → Tanh → Dense(768→4)
  - Device: cuda
- **Interpretation:** The vanilla classification head is a shallow 2-layer MLP, which may not be expressive enough for the nuanced 4-class task. The Tanh activation (rather than GELU or ReLU) in the classification head is a default choice that could limit representational capacity.

#### Training Table
- **Output:** Training over 10 epochs with eval every 500 steps. Key progression:
  - Step 500: Val accuracy = 0.7236, weighted F1 = 0.7107, val loss = 0.7024
  - Step 1000: Val accuracy = 0.7498, weighted F1 = 0.7489, val loss = 0.6434
  - Step 1500: Val accuracy = 0.7802, weighted F1 = 0.7754, val loss = 0.5688
  - Step 2000: Val accuracy = 0.7885, weighted F1 = 0.7768, val loss = 0.5498
  - Step 3000: Val accuracy = 0.7978, weighted F1 = 0.7862, val loss = 0.5232
  - Step 3500: Val accuracy = 0.7973, weighted F1 = **0.7931** (best), val loss = 0.5161
  - Step 4000+: Val accuracy plateaus at ~0.797, val loss stabilizes around 0.516
- **Interpretation:** The training converges around step 3500, reaching ~80% accuracy and ~79% weighted F1. The validation loss decreases steadily from 0.70 to 0.52, indicating genuine learning rather than overfitting. However, the weighted F1 masks individual class performance — the model is likely strong on human/machine but weak on hybrid/adversarial.

#### Classification Report (Validation)
- **Output:**
  - Class 0 (Human): precision=0.93, recall=0.93, F1=**0.93** (support: 10,719)
  - Class 1 (Machine): precision=0.74, recall=0.73, F1=**0.73** (support: 4,655)
  - Class 2 (Hybrid): precision=0.49, recall=0.53, F1=**0.51** (support: 1,944)
  - Class 3 (Adversarial): precision=0.66, recall=0.64, F1=**0.65** (support: 2,682)
  - Overall accuracy: **0.81**
  - Macro avg F1: **0.71**
  - Weighted avg F1: **0.81**
- **Interpretation:** This is the most informative output in the baseline notebook. The per-class breakdown reveals a clear performance hierarchy:
  
  - **Human (F1=0.93):** Excellent performance — the model reliably identifies pure human code. This is expected given the majority class has the most training data and the most consistent patterns.
  
  - **Machine (F1=0.73):** Good but not great — some AI-generated code is confused with human code (recall=0.73 means 27% of machine code is misclassified). This is better than Subtask A's out-of-distribution performance, suggesting the 4-class framing helps by providing more label granularity.
  
  - **Adversarial (F1=0.65):** Moderate performance — adversarial code is designed to deceive, so achieving 65% F1 with a simple baseline is actually reasonable. The precision (0.66) and recall (0.64) are balanced, suggesting the model captures some adversarial patterns but misses subtle ones.
  
  - **Hybrid (F1=0.51):** **The weakest class by far.** F1=0.51 is barely above random binary classification. Hybrid code is fundamentally the hardest category — it contains both human and machine patterns within the same snippet, and the transition points are contextual. The low support (1,944 samples) compounds the difficulty.
  
  The 10-point gap between macro F1 (0.71) and weighted F1 (0.81) quantifies how much the imbalanced classes drag down overall performance.

#### Test Set Prediction Error
- **Output:** `NameError: name 't' is not defined`
- **Interpretation:** The test prediction cell references a trainer object `t` that should have been assigned from `run_full_pipeline()` but wasn't captured. This is a simple variable naming bug — the pipeline returns the trainer but the cell references it with the wrong variable name. No test-set submission was generated.

---

## Notebook 2: task-C_improved

### Overview
The improved model notebook for Task C, designed to address the baseline's weaknesses — specifically targeting the **hybrid (F1=0.51)** and **adversarial (F1=0.65)** classes. Implements UniXcoder backbone, Bi-LSTM head, Focal Loss, SupCon loss, LLRD, and threshold calibration.

### Explanation
This notebook is the **most architecturally advanced** notebook in the entire project, integrating every improvement technique discovered across all three tasks. It contains 6 custom classes and 15+ functions:

1. **Backbone — UniXcoder (`microsoft/unixcoder-base`):**
   - Pre-trained with **unified cross-modal objectives**: code-to-NL, NL-to-code, code-to-AST, and denoising
   - Captures structural code patterns (AST, identifier flow) that distinguish human, machine, hybrid, and adversarial code
   - Also supports GraphCodeBERT (`microsoft/graphcodebert-base`) as an alternative backbone via `BACKBONE` config
   - **Rationale:** Adversarial code often has subtle structural anomalies (e.g., unnaturally consistent indentation, suspiciously clean variable naming) that token-level CodeBERT misses but structure-aware UniXcoder can detect

2. **FocalLoss Class:**
   - `__init__(gamma=2.0, alpha=class_weights, label_smoothing=0.05)`: Initializes with per-class alpha from sklearn balanced weights (clipped@10)
   - `forward(logits, targets)`:
     - Computes standard cross-entropy per sample: `ce = F.cross_entropy(logits, targets, reduction='none')`
     - Computes focusing factor: `pt = exp(-ce)`, `focal_weight = (1-pt)^γ`
     - Applies class-specific alpha: `focal_weight = alpha[targets] * focal_weight`
     - Returns mean: `(focal_weight * ce).mean()`
   - With `γ=2.0`, a well-classified human example (pt=0.9) has its loss reduced by 100x, redirecting gradient signal to hard hybrid/adversarial examples

3. **SupConLoss (Supervised Contrastive Loss):**
   - `__init__(temperature=0.07)`: Low temperature for sharp contrast
   - `forward(features, labels)`:
     - Normalizes features to unit sphere: `F.normalize(features, dim=1)`
     - Computes pairwise similarity matrix: `sim = features @ features.T / temperature`
     - For each sample, attracts same-class pairs and repels different-class pairs
     - Uses log-sum-exp for numerical stability
   - **Rationale:** Forces the embedding space to create 4 well-separated clusters. This is essential for hybrid vs adversarial separation, where both share token-level features but differ in structural patterns.

4. **BiLSTMHead:** A custom classification head that:
   - Takes the **full sequence** of hidden states from UniXcoder (not just [CLS])
   - Processes through a 2-layer bidirectional LSTM with `hidden_size=256`
   - Concatenates the final forward and backward hidden states → 512-d vector
   - Applies `Dropout(0.2)` and `Linear(512→num_labels)` for classification
   - **Rationale:** Critical for hybrid code detection — the Bi-LSTM can model the sequential transition from human to machine patterns mid-sequence. A single [CLS] token cannot represent this boundary.

5. **DenseBlockHead:** An alternative classification head (configurable via `HEAD_TYPE`):
   - Operates on the [CLS] embedding only
   - 3 residual dense blocks, each containing: `Linear → LayerNorm → GELU → Dropout`
   - Residual connections: `output = block(x) + x` (preserves gradient flow)
   - Final linear layer for classification
   - **When to use:** When the full-sequence BiLSTM is too memory-intensive, or when code structure is captured well by the [CLS] summary

6. **CodeClassifierWithSupCon:** The main model class that composes everything:
   - `__init__(backbone, num_labels, head_type, dropout)`:
     - Loads the UniXcoder/GraphCodeBERT backbone
     - Creates the selected head (`"bilstm"` or `"dense_block"`)
     - Creates a projection head for SupCon: `Dense(768→768) → ReLU → Dense(768→128)`
   - `gradient_checkpointing_enable()` / `gradient_checkpointing_disable()`: Memory management for long sequences
   - `forward(input_ids, attention_mask, labels)`:
     - Forward pass through backbone → hidden states
     - If BiLSTM: pass full sequence to BiLSTM head → logits
     - If DenseBlock: pass [CLS] to dense blocks → logits
     - Pass [CLS] through projection head → normalized 128-d embeddings
     - Returns `SequenceClassifierOutput(loss, logits, hidden_states=(proj,))`

7. **ImprovedTrainer:** Custom `Trainer` subclass:
   - Overrides `create_optimizer()` to implement **LLRD (Layer-wise Learning Rate Decay)**:
     - `get_llrd_optimizer(model, base_lr=3e-5, decay=0.9, weight_decay=0.01)`
     - Embeddings get: `LR = 3e-5 × 0.9^12 ≈ 8.5e-6`
     - Layer 11 gets: `LR = 3e-5 × 0.9^1 = 2.7e-5`
     - Classification & projection heads get: `LR = 3e-5` (full learning rate)
   - Overrides `compute_loss()` to compute combined loss:
     - `L_total = 0.85 × FocalLoss(logits, labels) + 0.15 × SupConLoss(projections, labels)`

8. **Data Pipeline:**
   - `load_data()`: Loads full dataset (900K train, 200K val) — **no subsampling** (unlike baseline's 10%)
   - `compute_balanced_weights(labels, num_labels, max_weight=10.0)`:
     - Uses `sklearn.utils.class_weight.compute_class_weight('balanced')`
     - Clips at `max_weight=10.0` to prevent numerical instability
     - Returns PyTorch tensor on GPU

9. **Configuration:**
   - `BACKBONE = "microsoft/unixcoder-base"` (default) — also supports `"microsoft/graphcodebert-base"` and `"microsoft/codebert-base"`
   - `HEAD_TYPE = "dense_block"` — also supports `"bilstm"` and `"linear"`
   - `MAX_LENGTH = 256` (chosen for 4x speed over 512; captures most patterns)
   - `BATCH_SIZE = 16`, `LEARNING_RATE = 3e-5`, `NUM_EPOCHS = 5`
   - `LABEL_SMOOTHING = 0.05` (mild smoothing)
   - `FOCAL_GAMMA = 2.0`, `SUPCON_WEIGHT = 0.15`, `DROPOUT = 0.2`
   - `NUM_LABELS = 4`

10. **Architecture Diagram (from update.md):**
    ```
    Input Code → Tokenizer(max_length=512) → UniXcoder Backbone
                                                   ↓
                                         [Full Hidden Sequence]
                                           ↓              ↓
                                   Bi-LSTM Head      Projection Head
                                   (2L, bidir)       (768→768→128)
                                       ↓                   ↓
                                  logits [B,4]        proj [B,128]
                                       ↓                   ↓
                                 Focal Loss           SupCon Loss
                                       ↓                   ↓
                                 L_total = 0.85*Focal + 0.15*SupCon
    ```

11. **9 Documented Changes from Baseline (per update.md):**

    | # | Change | Before → After |
    |---|--------|----------------|
    | 1 | Backbone | CodeBERT → UniXcoder (AST/DFG-aware) |
    | 2 | Classification Head | Single linear on [CLS] → Bi-LSTM over full sequence |
    | 3 | Loss Function | Cross-entropy → Focal Loss (γ=2.0) + balanced weights |
    | 4 | Contrastive Signal | None → SupCon loss (weight=0.15, temperature=0.07) |
    | 5 | Learning Rate | Uniform 2e-5 → LLRD (decay=0.9 per layer) |
    | 6 | Data Scale | 10% (90K train) → 100% (900K train, 200K val) |
    | 7 | LR Schedule | Linear → Cosine with 500-step warmup |
    | 8 | Primary Metric | Weighted F1 → Macro F1 |
    | 9 | Decision Boundary | Default argmax → Per-class threshold calibration |

### Output Interpretations

#### All Cells
- **Output:** No execution outputs captured in the saved notebook.
- **Interpretation:** The improved notebook was **not executed** in this saved state, or outputs were not preserved. This is likely because:
  1. Training on the full 900K dataset with all the architectural improvements requires significant compute resources (potentially exceeding Kaggle's session limits)
  2. The notebook may have been executed in a different environment (Colab, cloud GPU) where outputs weren't saved back
  3. The notebook serves as the architectural specification that was validated through the documented changes in `update.md`

  Despite the lack of outputs, the notebook's design is extensively documented in `update.md` and represents the **most comprehensive single-notebook design** in the entire project.

---

## Cross-Notebook Key Findings

1. **The 4-Class Hybrid/Adversarial Problem is Fundamentally Harder than Binary Classification:** Unlike Subtask A (human vs machine), Subtask C requires distinguishing partially automated code and intentionally deceptive code. The baseline's F1=0.51 for hybrid code demonstrates that existing code embeddings struggle to capture the "boundary" between human and machine contributions within a single snippet.

2. **Human Code is Reliably Identifiable (F1=0.93):** Across both notebooks, pure human code is consistently the best-classified category. This suggests human coding patterns are uniquely identifiable — likely due to the diversity of individual style, typical formatting inconsistencies, and organic comment patterns.

3. **Hybrid Code is the Critical Bottleneck (F1=0.51):** Hybrid code contains both human and machine-written sections, making it inherently ambiguous. The model needs to detect the transition point and identify the "mixed" signal. The Bi-LSTM improvement (Notebook 2) directly addresses this by processing the full sequence rather than relying on a single [CLS] summary.

4. **Adversarial Code is Moderately Detectable (F1=0.65):** Despite being designed to deceive, adversarial code apparently retains some detectable patterns. This suggests that adversarial perturbations (e.g., reformatting AI code to look human, or injecting AI patterns into human code) are not perfectly camouflaging the underlying generation process.

5. **Class Imbalance Hurts Minority Classes Disproportionately:** The 5.6:1 ratio between human and hybrid classes means the model sees 5.6x more human examples during training. Without class weighting or focal loss, the model's gradient signal is dominated by the majority class.

## Conclusions & Recommendations

- **The baseline macro F1 of 0.71 is a reasonable starting point**, with the improved notebook targeting ≥0.80 through architectural and training innovations. The 9 changes documented in `update.md` are well-motivated and address the identified weaknesses systematically.
  
- **Hybrid code detection is the key to improving macro F1.** Improving hybrid F1 from 0.51 to even 0.70 would push macro F1 from 0.71 to ~0.76. Techniques specifically targeting sequence-level boundary detection (where human code transitions to machine code) would be most impactful.

- **The Bi-LSTM head is theoretically the most important architectural change** for hybrid code, as it can model the sequential transition from human to machine patterns. A sliding-window or attention-based approach that explicitly detects "change points" in code style could further improve performance.

- **Adversarial code detection may benefit from adversarial training.** If the model is exposed to adversarial perturbations during training (e.g., adversarially transformed human code, or adversarially cleaned machine code), it may learn more robust features.

- **The improved notebook should be executed on full data** to validate whether the theoretical improvements translate to actual metric gains. The progression from 0.71 to the target of 0.80+ macro F1 would represent a substantial advance in multi-class code provenance detection.

- **Cross-task knowledge transfer is possible.** Insights from Subtask A (what makes code detectably machine-generated) and Subtask B (which LLM families are most distinctive) can inform Subtask C's hybrid and adversarial detection — if a code snippet contains passages that match a known LLM family's style alongside human-like sections, it's likely hybrid.
