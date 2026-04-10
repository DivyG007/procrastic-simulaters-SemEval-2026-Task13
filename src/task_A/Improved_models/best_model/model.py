"""
CodeBERT model with a deep classification head.
"""

import torch
import torch.nn as nn
from transformers import RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput


class DeepHeadCodeBERT(nn.Module):
    """
    CodeBERT [CLS] (768-d) concatenated with handcrafted features
    \u2192 deep MLP head: input_dim \u2192 256 \u2192 128 \u2192 64 \u2192 num_labels
    Uses GELU activations, LayerNorm, and moderate dropout.
    """

    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        num_labels: int = 2,
        num_features: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_labels = num_labels

        # --- Transformer backbone ---
        self.transformer = RobertaModel.from_pretrained(model_name)
        self.config = self.transformer.config
        hidden_size = self.config.hidden_size  # 768

        # --- Feature normalisation ---
        # Added normalization for handcrafted features
        self.feat_norm = nn.LayerNorm(num_features)

        # --- Deep classification head ---
        input_dim = hidden_size + num_features  # 776
        self.head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(64, num_labels),
        )

        # Initialise head weights (Xavier)
        self._init_weights()

    def _init_weights(self):
        """Initialize the classification head weights."""
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        code_features=None,
        **kwargs,
    ):
        """Forward pass for the model."""
        out = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask
        )
        # Take [CLS] token embedding (first token)
        cls_vec = out.last_hidden_state[:, 0, :]  # (Batch, 768)

        if code_features is not None:
            # Normalize and concatenate features
            feat = self.feat_norm(code_features.float())
            combined = torch.cat([cls_vec, feat], dim=-1)  # (Batch, 776)
        else:
            # Fallback if no features provided
            combined = torch.cat(
                [cls_vec,
                 cls_vec.new_zeros(cls_vec.size(0), self.feat_norm.normalized_shape[0])],
                dim=-1,
            )

        logits = self.head(combined)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)
