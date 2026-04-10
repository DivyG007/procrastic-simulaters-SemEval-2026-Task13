"""Model definitions and optimizer helpers for Task B improved GraphCodeBERT."""

import numpy as np
import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel, RobertaPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from config import Config


class RobertaForClassificationWithSupCon(RobertaPreTrainedModel):
    """RoBERTa encoder with classification and projection heads."""

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        hidden = config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(hidden, config.num_labels),
        )
        self.projector = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 128),
        )
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """Forward pass that returns logits and projection embeddings."""
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_emb)
        proj = self.projector(cls_emb)
        return SequenceClassifierOutput(logits=logits, hidden_states=(proj,))


def build_model(cfg: Config):
    """Load custom GraphCodeBERT model with task label count."""
    config = RobertaConfig.from_pretrained(cfg.model_name, num_labels=11)
    model = RobertaForClassificationWithSupCon.from_pretrained(cfg.model_name, config=config)
    return model


def get_llrd_optimizer(model, cfg: Config):
    """Create AdamW optimizer with layer-wise learning-rate decay."""
    opt_params = []
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]

    num_layers = model.config.num_hidden_layers
    for layer_idx in range(num_layers):
        lr = cfg.learning_rate * (cfg.llrd_decay ** (num_layers - layer_idx))
        layer_params = []
        layer_params_no_decay = []
        layer_name = f"roberta.encoder.layer.{layer_idx}."

        for name, param in model.named_parameters():
            if layer_name in name:
                if any(nd in name for nd in no_decay):
                    layer_params_no_decay.append(param)
                else:
                    layer_params.append(param)

        if layer_params:
            opt_params.append({"params": layer_params, "lr": lr, "weight_decay": cfg.weight_decay})
        if layer_params_no_decay:
            opt_params.append({"params": layer_params_no_decay, "lr": lr, "weight_decay": 0.0})

    embed_lr = cfg.learning_rate * (cfg.llrd_decay ** (num_layers + 1))
    embed_params = [p for n, p in model.named_parameters() if "roberta.embeddings" in n]
    if embed_params:
        opt_params.append({"params": embed_params, "lr": embed_lr, "weight_decay": cfg.weight_decay})

    head_params = [p for n, p in model.named_parameters() if "classifier" in n or "projector" in n]
    if head_params:
        opt_params.append({"params": head_params, "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay})

    return torch.optim.AdamW(opt_params)


def mixup_data(embeddings, labels, alpha: float = 0.4):
    """Embedding-space mixup helper retained for notebook parity."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = embeddings.size(0)
    index = torch.randperm(batch_size, device=embeddings.device)
    mixed_emb = lam * embeddings + (1 - lam) * embeddings[index]
    return mixed_emb, labels, labels[index], lam
