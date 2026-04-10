"""Training and evaluation pipeline for Task B improved GraphCodeBERT."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from transformers import (
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    RobertaTokenizer,
    Trainer,
    TrainingArguments,
)

from config import Config, ID_TO_LABEL, NUM_LABELS
from losses import FocalLoss, SupConLoss
from model import build_model, get_llrd_optimizer


class ImprovedTrainer(Trainer):
    """Custom trainer with FocalLoss + SupConLoss objective."""

    def __init__(self, cfg: Config, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        alpha = class_weights if class_weights is not None else None
        self.cfg = cfg
        self.focal_loss = FocalLoss(alpha=alpha, gamma=cfg.focal_gamma)
        self.supcon_loss = SupConLoss(temperature=cfg.supcon_temperature)

    def create_optimizer(self):
        """Create optimizer using layer-wise LR decay."""
        self.optimizer = get_llrd_optimizer(self.model, self.cfg)
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute weighted combination of focal and supervised contrastive loss."""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        proj = outputs.hidden_states[0] if outputs.hidden_states else None

        loss_ce = self.focal_loss(logits, labels)
        loss_supcon = torch.tensor(0.0, device=logits.device)
        if proj is not None and self.cfg.supcon_weight > 0:
            loss_supcon = self.supcon_loss(proj, labels)

        loss = (1 - self.cfg.supcon_weight) * loss_ce + self.cfg.supcon_weight * loss_supcon
        return (loss, outputs) if return_outputs else loss


@dataclass
class RunArtifacts:
    """Container for training outputs needed in later stages."""

    trainer: Trainer
    val_dataset: object


class GraphCodeBERTTrainerB:
    """End-to-end trainer wrapper for Subtask B."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tokenizer = None
        self.model = None
        self.class_weights = None
        self.tag = cfg.model_name.split("/")[-1]

    def initialize_model_and_tokenizer(self) -> None:
        """Initialize tokenizer and model."""
        self.tokenizer = RobertaTokenizer.from_pretrained(self.cfg.model_name)
        self.model = build_model(self.cfg)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

    def compute_metrics(self, eval_pred):
        """Compute task metrics with per-class logging."""
        predictions, labels = eval_pred
        preds = np.argmax(predictions, axis=1)

        accuracy = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)

        _, _, f1_per_class, _ = precision_recall_fscore_support(
            labels, preds, average=None, zero_division=0, labels=range(NUM_LABELS)
        )
        class_f1_str = " | ".join(f"{ID_TO_LABEL[i][:3]}={f1_per_class[i]:.2f}" for i in range(NUM_LABELS))
        print(f"Per-class F1: {class_f1_str}")

        return {"macro_f1": macro_f1, "weighted_f1": weighted_f1, "accuracy": accuracy}

    def train(self, train_dataset, val_dataset) -> Trainer:
        """Run Hugging Face training loop."""
        steps_per_epoch = max(1, len(train_dataset) // (self.cfg.batch_size * self.cfg.grad_accum_steps))
        eval_save_steps = max(200, steps_per_epoch // 3)

        args = TrainingArguments(
            output_dir=self.cfg.output_dir,
            num_train_epochs=self.cfg.num_epochs,
            per_device_train_batch_size=self.cfg.batch_size,
            per_device_eval_batch_size=self.cfg.batch_size * 2,
            gradient_accumulation_steps=self.cfg.grad_accum_steps,
            warmup_steps=self.cfg.warmup_steps,
            weight_decay=self.cfg.weight_decay,
            logging_dir=f"{self.cfg.output_dir}/logs",
            logging_steps=50,
            evaluation_strategy="steps",
            eval_steps=eval_save_steps,
            save_strategy="steps",
            save_steps=eval_save_steps,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            remove_unused_columns=False,
            learning_rate=self.cfg.learning_rate,
            lr_scheduler_type="cosine",
            save_total_limit=5,
            fp16=self.cfg.fp16,
            report_to="none",
        )

        def _preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                return logits[0]
            return logits

        trainer = ImprovedTrainer(
            cfg=self.cfg,
            class_weights=self.class_weights,
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            compute_metrics=self.compute_metrics,
            preprocess_logits_for_metrics=_preprocess_logits_for_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.cfg.early_stopping_patience)],
        )

        trainer.train()
        trainer.save_model()
        self.tokenizer.save_pretrained(self.cfg.output_dir)
        return trainer

    def evaluate_model(self, trainer: Trainer, val_dataset) -> None:
        """Print validation report and confusion matrix."""
        predictions = trainer.predict(val_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids

        target_names = [ID_TO_LABEL[i] for i in range(NUM_LABELS)]
        print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

        cm = confusion_matrix(y_true, y_pred, labels=range(NUM_LABELS))
        print(pd.DataFrame(cm, index=target_names, columns=target_names))

        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        print(f"Macro F1: {macro_f1:.4f}")

    def run_full_pipeline(self, train_dataset, val_dataset) -> RunArtifacts:
        """Run train and evaluation stages."""
        self.initialize_model_and_tokenizer()
        trainer = self.train(train_dataset, val_dataset)
        self.evaluate_model(trainer, val_dataset)
        return RunArtifacts(trainer=trainer, val_dataset=val_dataset)
