# Update Log — Improved Task C Notebook

**File Modified:** `Improved_Models/task_C/task-C.ipynb`

**Date:** 2026-04-06

**Goal:** Improve macro-F1 from 0.71 → ≥ 0.80 on SemEval-2026 Task 13 Subtask C (4-class hybrid code detection), specifically targeting the weak **hybrid** (F1=0.51) and **adversarial** (F1=0.65) classes.

---

## Summary of Root Causes Addressed

| Problem | Root Cause | Fix Applied |
|---------|-----------|-------------|
| Weak hybrid/adversarial recall | No class weighting in loss | Focal Loss + sklearn balanced weights (clipped@10) |
| Shallow decision boundary | Single linear head on [CLS] | Bi-LSTM head over full hidden sequence |
| CodeBERT misses code structure | No AST/DFG-aware pretraining | UniXcoder backbone (+ GraphCodeBERT option) |
| Only 10% of data used | `sample_fraction=0.1` hardcoded | Full dataset (900K train, 200K val) |
| max_length=128 truncates code | AI boilerplate/adversarial patterns cut off | max_length=512 with gradient checkpointing |
| Uniform learning rate | All layers trained at same speed | LLRD: decay=0.9 per layer |
| No embedding-level separation | Classifier alone doesn't separate classes | SupCon auxiliary loss (weight=0.15) |
| Default argmax biased to majority | No threshold calibration | Per-class threshold sweep post-training |
| Weighted F1 as metric | Doesn't reflect minority performance | macro_f1 as primary metric |

---

## Change 1: Backbone — UniXcoder (Default) / GraphCodeBERT

- **Before:** `microsoft/codebert-base` — token-level pretraining only
- **After:** `microsoft/unixcoder-base` — unified cross-modal pretraining with AST/code structure
- **Also available:** `microsoft/graphcodebert-base` — data-flow-graph aware
- **Rationale:** UniXcoder captures structural code patterns (AST, identifier flow) that distinguish human, machine, hybrid, and adversarial code. Adversarial code often has subtle structural anomalies that token-level CodeBERT misses.

---

## Change 2: Bi-LSTM Classification Head

- **Before:** `RobertaForSequenceClassification` — single linear head on [CLS]
- **After:** `BiLSTMHead` — 2-layer bidirectional LSTM over the full hidden sequence, classifying from concatenated forward/backward final states
- **Also available:** `DenseBlockHead` — 3 residual dense blocks with LayerNorm + GELU
- **Rationale:** The Bi-LSTM captures positional and sequential patterns across the entire code embedding sequence, not just the [CLS] summary. This is critical for hybrid code where the transition from human to LLM-generated tokens occurs mid-sequence.

---

## Change 3: Focal Loss + Class Weighting

- **Before:** Standard `nn.CrossEntropyLoss` (no weighting)
- **After:** Custom `FocalLoss(gamma=2.0, alpha=class_weights, label_smoothing=0.05)`
- **Class weights:** sklearn balanced, clipped at 10.0
- **Rationale:** With 485K human vs 85K hybrid, the model is overwhelmed by easy human examples. Focal loss reduces their gradient contribution by ~100x when well-classified (pt=0.9), while class weighting ensures rare classes get proportionally stronger signal.

---

## Change 4: Supervised Contrastive Loss

- **Before:** No embedding-level supervision
- **After:** `SupConLoss(temperature=0.07)` on a dedicated projection head (768→768→128)
- **Total loss:** `L = 0.85 * Focal + 0.15 * SupCon`
- **Rationale:** Forces the model to create well-separated embedding clusters per class. This is essential for hybrid vs adversarial separation, where both overlap significantly in token-level features but differ in structural patterns.

---

## Change 5: Layer-wise Learning Rate Decay

- **Before:** Uniform LR=2e-5 for all parameters
- **After:** LLRD with decay=0.9: embeddings get LR≈8.5e-6, layer 11 gets 3e-5, heads get 3e-5
- **Rationale:** Lower transformer layers capture general language/code patterns that should be preserved. Upper layers and heads need aggressive adaptation for the 4-class task.

---

## Change 6: Full Dataset + max_length 512

- **Before:** 10% sample (90K train), max_length=128
- **After:** 100% data (900K train, 200K val), max_length=512
- **Memory management:** batch_size=8, grad_accum=4 (effective batch=32), gradient checkpointing enabled
- **Rationale:** Adversarial and hybrid patterns are subtle and require both more data and longer context windows. LLM-generated adversarial code often has distinctive patterns beyond token 128.

---

## Change 7: Cosine LR Schedule

- **Before:** Linear schedule
- **After:** Cosine schedule with 500-step warmup
- **Rationale:** Cosine is more stable for longer training runs, providing gentler LR reduction that helps with minority class convergence.

---

## Change 8: Macro-F1 as Primary Metric

- **Before:** Weighted F1 (masks poor minority performance)
- **After:** Macro F1 (equal weight to all 4 classes)
- **Also reports:** Per-class precision/recall/F1, confusion matrix

---

## Change 9: Per-Class Threshold Calibration

- **New cell added post-training**
- Sweeps thresholds [0.05, 0.95] per class on validation set
- Applies calibrated thresholds at inference time
- Reports calibrated vs uncalibrated macro-F1 comparison

---

## Architecture Diagram

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

---

## How to Run Ablations

### Backbone experiments
```python
BACKBONE = "microsoft/codebert-base"        # baseline
BACKBONE = "microsoft/graphcodebert-base"    # +DFG awareness
BACKBONE = "microsoft/unixcoder-base"        # +unified cross-modal (default)
```

### Head experiments
```python
HEAD_TYPE = "linear"       # simple dense head
HEAD_TYPE = "bilstm"       # Bi-LSTM over full hidden sequence (default)
HEAD_TYPE = "dense_block"  # 3 residual dense blocks on [CLS]
```
