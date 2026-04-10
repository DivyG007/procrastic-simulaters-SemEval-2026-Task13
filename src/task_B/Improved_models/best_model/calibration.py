"""Per-class threshold calibration utilities."""

import numpy as np
from scipy.special import softmax
from sklearn.metrics import f1_score


def calibrate_thresholds(logits: np.ndarray, y_true: np.ndarray, num_labels: int) -> np.ndarray:
    """Find per-class thresholds that maximize one-vs-rest F1."""
    probs = softmax(logits, axis=1)
    thresholds = np.zeros(num_labels)

    for class_idx in range(num_labels):
        binary_true = (y_true == class_idx).astype(int)
        class_probs = probs[:, class_idx]

        best_f1, best_threshold = 0.0, 0.5
        for t in np.arange(0.05, 0.95, 0.02):
            preds = (class_probs >= t).astype(int)
            if preds.sum() == 0:
                continue
            score = f1_score(binary_true, preds, zero_division=0)
            if score > best_f1:
                best_f1, best_threshold = score, t

        thresholds[class_idx] = best_threshold

    return thresholds


def predict_with_thresholds(logits: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Apply calibrated thresholds and fallback to argmax when needed."""
    probs = softmax(logits, axis=1)
    preds = np.full(len(logits), -1, dtype=int)

    for i in range(len(logits)):
        above = np.where(probs[i] >= thresholds)[0]
        if len(above) > 0:
            preds[i] = above[np.argmax(probs[i][above])]
        else:
            preds[i] = np.argmax(probs[i])

    return preds
