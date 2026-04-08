# Update Log — GraphCodeBERT Task B Notebook

**File Modified:** `Improved_Models/task_B/graphcodebert_task_b.ipynb`

**Date:** 2026-04-06

**Goal:** Improve macro F1 from 0.43 → ≥ 0.65 on SemEval-2026 Task 13 Subtask B (11-class code authorship attribution)

---

## Summary of Root Causes Addressed

| Problem | Root Cause | Fix Applied |
|---------|-----------|-------------|
| Model biased toward 'human' | 8158/10001 val predictions are class 0 | Focal Loss + aggressive class weighting |
| Weak minority recall | sqrt-scaled weights too conservative | sklearn balanced weights, clipped@8 |
| Truncated code patterns | max_length=256 misses AI boilerplate | Increased to 512 |
| fp16 disabled unnecessarily | P100 fallback was wrong — T4 supports fp16 | Enabled fp16=True |
| Label smoothing hurts minorities | 0.1 smoothing blunts low-support confidence | Reduced to 0.0 |
| No threshold calibration | Default argmax favors majority class | Per-class threshold sweep added |
| No contrastive signal | Classifier alone doesn't separate embeddings | SupConLoss auxiliary head added |

---

## Change 1: Aggressive Class Weighting

**Cell 5 (utilities) + Cell 6 (trainer class)**

- **Before:** `compute_class_weights()` with `method="sqrt"` — sqrt of inverse frequency, ~15:1 ratio
- **After:** `compute_balanced_weights()` using `sklearn.utils.class_weight.compute_class_weight('balanced')`, clipped at `MAX_CLASS_WEIGHT=8`
- **Rationale:** sqrt scaling was too conservative for extreme imbalance (human: 442K vs gemma: 1.9K). Balanced weights with clipping gives stronger pull on minorities without instability.

```python
# BEFORE
def compute_class_weights(label_counts, num_labels, method="sqrt"):
    raw_weight = total / (num_labels * count)
    w = np.sqrt(raw_weight)  # too conservative

# AFTER
def compute_balanced_weights(labels, num_labels, max_weight=8.0):
    weights = compute_class_weight('balanced', classes=np.arange(num_labels), y=labels)
    weights = np.clip(weights, 1.0, max_weight)
```

---

## Change 2: Threshold Calibration (New Cell 8)

**Entirely new cell added**

- **Before:** No threshold calibration — argmax always favors majority class
- **After:** After training, sweeps per-class decision thresholds (0.05 to 0.95, step 0.02) on the validation set to maximize per-class binary F1. Applies calibrated thresholds at inference time.
- **Functions added:** `calibrate_thresholds()`, `predict_with_thresholds()`
- **Reports:** Calibrated vs uncalibrated macro F1, calibrated confusion matrix

---

## Change 3: max_length 256 → 512

**Cell 3 (configuration)**

- **Before:** `MAX_LENGTH = 256`, `BATCH_SIZE = 16`, `GRAD_ACCUM_STEPS = 2`
- **After:** `MAX_LENGTH = 512`, `BATCH_SIZE = 8`, `GRAD_ACCUM_STEPS = 4`
- **Rationale:** AI-generated code has distinctive boilerplate/comment patterns beyond token 256. Reduced batch size to fit in T4 VRAM; increased grad accumulation to keep effective batch=32.

```diff
- MAX_LENGTH         = 256
- BATCH_SIZE         = 16
- GRAD_ACCUM_STEPS   = 2
+ MAX_LENGTH         = 512
+ BATCH_SIZE         = 8
+ GRAD_ACCUM_STEPS   = 4
```

---

## Change 4: Focal Loss

**Cell 5 (new FocalLoss class)**

- **Before:** `nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)` in `WeightedTrainer`
- **After:** Custom `FocalLoss(gamma=2, alpha=class_weights)` in `ImprovedTrainer`
- **Rationale:** Focal loss down-weights easy 'human' examples (high pt), focusing gradient on hard-to-classify minority samples. With γ=2, the loss for a well-classified example (pt=0.9) is reduced by 100x.

```python
class FocalLoss(nn.Module):
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_weight = alpha_t * focal_weight
        return (focal_weight * ce_loss).mean()
```

---

## Change 5: Layer-wise Learning Rate Decay (LLRD)

**Cell 5 (new `get_llrd_optimizer()` function)**

- **Before:** Uniform learning rate (2e-5) for all layers
- **After:** Per-layer LR decay: `lr_layer_k = base_lr * (0.9 ** (12 - k))`. Embeddings get lowest LR, classifier/projector heads get highest.
- **Rationale:** Lower layers capture general language patterns that shouldn't change much. Higher layers + heads need more adaptation for the authorship task.

```python
def get_llrd_optimizer(model, base_lr=3e-5, decay=0.9, weight_decay=0.01):
    for layer_idx in range(num_layers):
        lr = base_lr * (decay ** (num_layers - layer_idx))
        # group params by layer with appropriate lr
```

Overridden in `ImprovedTrainer.create_optimizer()`.

---

## Change 6: Embedding-Space Mixup

**Cell 5 (new `mixup_data()` function)**

- **Before:** No data augmentation
- **After:** `mixup_data(embeddings, labels, alpha=0.4)` — interpolates embeddings using λ ~ Beta(0.4,0.4), mixes corresponding labels
- **Config:** `MIXUP_ALPHA = 0.4`

```python
def mixup_data(embeddings, labels, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(batch_size, device=embeddings.device)
    mixed_emb = lam * embeddings + (1 - lam) * embeddings[index]
    return mixed_emb, labels, labels[index], lam
```

---

## Change 7: Label Smoothing Removed

**Cell 3 (configuration)**

- **Before:** `LABEL_SMOOTHING = 0.1`
- **After:** `LABEL_SMOOTHING = 0.0`
- **Rationale:** Label smoothing distributes probability mass to incorrect classes. For minority classes with few samples, this blunts the model's confidence and reduces recall. With Focal Loss already handling the easy/hard example balance, smoothing is unnecessary.

---

## Change 8: Contrastive Loss Auxiliary Head (SupConLoss)

**Cell 5 (new classes)**

- **Before:** Standard `RobertaForSequenceClassification` with classification head only
- **After:** Custom `RobertaForClassificationWithSupCon` with:
  - Classification head (Dense → Tanh → Dense → num_labels)
  - Projection head (Dense → ReLU → Dense → 128) for contrastive loss
- **Loss:** `L_total = 0.8 * FocalLoss + 0.2 * SupConLoss`

```python
class SupConLoss(nn.Module):
    """Supervised contrastive loss on normalized [CLS] embeddings."""
    def forward(self, features, labels):
        # Attract same-class pairs, repel different-class pairs
        sim = torch.matmul(F.normalize(features), F.normalize(features).T) / temperature
```

---

## Change 9: Checkpoint Saving on Macro F1

**Cell 6 (training args)**

- **Before:** `save_total_limit=2`
- **After:** `save_total_limit=5`, `metric_for_best_model="macro_f1"` (was already set, but now with more checkpoints retained)
- **Also changed:** `load_best_model_at_end=True` loads the checkpoint with highest macro F1

---

## Other Training Config Changes

```diff
- LEARNING_RATE      = 2e-5
+ LEARNING_RATE      = 3e-5

- NUM_EPOCHS         = 10
+ NUM_EPOCHS         = 6

- WARMUP_RATIO       = 0.1
+ WARMUP_STEPS       = 500

- EARLY_STOPPING_PATIENCE = 3
+ EARLY_STOPPING_PATIENCE = 2

- fp16 = False  # P100 (sm_60) not supported
+ fp16 = True   # T4 (sm_75) fully supports fp16
```

---

## New Cells Added

| Cell | Purpose |
|------|---------|
| Cell 8 (Threshold Calibration) | Sweeps per-class thresholds, reports calibrated macro F1 |

---

## Classes/Functions Renamed or Replaced

| Before | After |
|--------|-------|
| `WeightedTrainer` | `ImprovedTrainer` (Focal + SupCon + LLRD) |
| `compute_class_weights()` | `compute_balanced_weights()` (sklearn) |
| `RobertaForSequenceClassification` | `RobertaForClassificationWithSupCon` (custom model) |

---

## New Classes/Functions Added

| Name | Purpose |
|------|---------|
| `FocalLoss` | Focal loss with per-class alpha and gamma focusing |
| `SupConLoss` | Supervised contrastive loss on [CLS] projections |
| `RobertaForClassificationWithSupCon` | Custom RoBERTa with classifier + projector heads |
| `get_llrd_optimizer()` | Layer-wise learning rate decay optimizer |
| `mixup_data()` | Embedding-space mixup augmentation |
| `calibrate_thresholds()` | Per-class threshold optimization on val set |
| `predict_with_thresholds()` | Inference with calibrated thresholds |

---

## Bugfix: Inhomogeneous Array Shape Error (2026-04-06)

**Error:** `ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (2,) + inhomogeneous part.`

**Root Cause:** The custom model `RobertaForClassificationWithSupCon` returns `SequenceClassifierOutput(logits=logits, hidden_states=(proj,))`. During evaluation, the HuggingFace Trainer collects both `logits` (shape `[N, 11]`) and `hidden_states` (a tuple of shape `[N, 128]`) as a single `predictions` tuple. When numpy tries to stack them, the mismatched shapes cause the error.

**Fix:** Added `preprocess_logits_for_metrics` callback that strips `hidden_states` before predictions reach `compute_metrics`:

```python
def _preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        return logits[0]  # extract only the classification logits
    return logits

trainer = ImprovedTrainer(
    ...,
    preprocess_logits_for_metrics=_preprocess_logits_for_metrics,
)
```

This ensures `compute_metrics` only sees the logits tensor, not the projection embeddings.

---

## Evaluation Enhancements

- **Per-class F1** printed at every eval step (not just at end)
- **Confusion matrix** printed after final evaluation
- **Calibrated confusion matrix** printed after threshold calibration
- **Calibrated vs uncalibrated macro F1** comparison reported
- Test submission now uses calibrated thresholds
