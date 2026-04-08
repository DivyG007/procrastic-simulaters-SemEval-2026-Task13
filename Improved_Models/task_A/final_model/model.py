"""
model.py — DeepHead Classification Model
==========================================
Parameterised deep classification head that works with any RoBERTa-family
backbone (CodeBERT, GraphCodeBERT, UniXcoder).

Architecture:
    [CLS] embedding (768-d) ⊕ 8 stylometric features = 776
    → 256 → 128 → 64 → num_labels
    with GELU activations, LayerNorm, and dropout between layers.
"""

import torch
import torch.nn as nn
from transformers import RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput

from config import NUM_CODE_FEATURES


class DeepHeadModel(nn.Module):
    """
    RoBERTa backbone + deep MLP classification head.

    The backbone is specified via `model_name` — this allows the same
    class to be used for CodeBERT, GraphCodeBERT, and UniXcoder since
    all three share the RoBERTa architecture.

    Args:
        model_name:   HuggingFace model identifier (e.g. 'microsoft/codebert-base')
        num_labels:   number of output classes (2 for Task A)
        num_features: number of handcrafted stylometric features (8)
        dropout:      dropout rate applied between MLP layers
    """

    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        num_labels: int = 2,
        num_features: int = NUM_CODE_FEATURES,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_labels = num_labels

        # ── Transformer backbone ──
        self.transformer = RobertaModel.from_pretrained(model_name)
        self.config = self.transformer.config
        hidden_size = self.config.hidden_size  # 768 for all three backbones

        # ── Feature normalisation layer ──
        self.feat_norm = nn.LayerNorm(num_features)

        # ── Deep classification head: (768 + num_features) → 256 → 128 → 64 → num_labels ──
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

        # Xavier initialisation for head weights
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
        """
        Forward pass.

        Args:
            input_ids:      (B, L)  token IDs
            attention_mask: (B, L)  attention mask
            labels:         (B,)   ground truth labels (optional, for loss)
            code_features:  (B, 8) handcrafted stylometric features (optional)

        Returns:
            SequenceClassifierOutput with loss (if labels given) and logits.
        """
        # Get contextualised representations from the backbone
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_vec = out.last_hidden_state[:, 0, :]  # [CLS] token → (B, 768)

        # Concatenate stylometric features (or zeros if absent)
        if code_features is not None:
            feat = self.feat_norm(code_features.float())
            combined = torch.cat([cls_vec, feat], dim=-1)  # (B, 776)
        else:
            combined = torch.cat(
                [cls_vec, cls_vec.new_zeros(cls_vec.size(0), self.feat_norm.normalized_shape[0])],
                dim=-1,
            )

        logits = self.head(combined)

        # Compute loss when labels are provided (training / evaluation)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)
