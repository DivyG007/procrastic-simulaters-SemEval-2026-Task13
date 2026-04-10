"""Trainer wrapper for Task C baseline notebook conversion."""

import os

import numpy as np
import torch
from sklearn.metrics import classification_report
from transformers import (
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    Trainer,
    TrainingArguments,
)

from config import Config
from data_utils import load_data, tokenize_datasets
from metrics import compute_metrics


class CodeBERTTrainer:
    """Class-based baseline trainer equivalent to the Task C notebook."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.max_length = cfg.max_length
        self.model_name = cfg.model_name
        self.tokenizer = None
        self.model = None
        self.num_labels = None

    def load_and_prepare_data(self):
        """Load parquet data and apply baseline 10% subsampling."""
        train_df, val_df = load_data(
            train_path=self.cfg.train_path,
            val_path=self.cfg.val_path,
            sample_fraction=self.cfg.sample_fraction,
            random_seed=self.cfg.random_seed,
        )

        self.num_labels = train_df["label"].nunique()

        print(f"Using {self.cfg.sample_fraction * 100:.0f}% of training data: {len(train_df)} samples")
        print(f"Using {self.cfg.sample_fraction * 100:.0f}% of validation data: {len(val_df)} samples")
        print(f"Number of unique labels: {self.num_labels}")
        print(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}")

        return train_df, val_df

    def initialize_model_and_tokenizer(self):
        """Initialize tokenizer and `RobertaForSequenceClassification` model."""
        print(f"Initializing {self.model_name} model and tokenizer...")
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = RobertaForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            problem_type="single_label_classification",
        ).to(device)

    def prepare_datasets(self, train_df, val_df):
        """Tokenize and convert pandas dataframes to HuggingFace datasets."""
        return tokenize_datasets(train_df, val_df, self.tokenizer, self.max_length)

    def train(self, train_dataset, val_dataset):
        """Run HF Trainer loop with baseline settings."""
        os.makedirs(self.cfg.output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=self.cfg.output_dir,
            num_train_epochs=self.cfg.num_epochs,
            per_device_train_batch_size=self.cfg.batch_size,
            per_device_eval_batch_size=self.cfg.batch_size,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=5,
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            remove_unused_columns=False,
            learning_rate=self.cfg.learning_rate,
            lr_scheduler_type="linear",
            save_total_limit=2,
            report_to=[],
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        print("Start training")
        trainer.train()

        trainer.save_model()
        self.tokenizer.save_pretrained(self.cfg.output_dir)
        print(f"Training completed. Model saved to {self.cfg.output_dir}")
        return trainer

    def evaluate_model(self, trainer, val_dataset):
        """Print classification report from validation predictions."""
        predictions = trainer.predict(val_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids

        print("Classification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))
        return predictions

    def run_full_pipeline(self):
        """Execute data loading, training, and evaluation."""
        train_df, val_df = self.load_and_prepare_data()
        self.initialize_model_and_tokenizer()
        train_dataset, val_dataset = self.prepare_datasets(train_df, val_df)
        trainer = self.train(train_dataset, val_dataset)
        self.evaluate_model(trainer, val_dataset)
        print("Pipeline completed successfully!")
        return trainer
