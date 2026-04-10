"""
Custom Trainer implementation with Layer-wise Learning Rate Decay (LLRD).
"""

import torch
from transformers import Trainer, get_cosine_schedule_with_warmup


def get_layer_wise_lr_groups(
    model,
    base_lr: float = 2e-5,
    head_lr: float = 1e-3,
    weight_decay: float = 0.01,
    llrd_factor: float = 0.95,
):
    """
    Assign per-layer learning rates:
      - Embedding layer gets base_lr * llrd_factor^N (lowest LR)
      - Each encoder layer i gets base_lr * llrd_factor^(N-i)
      - Classification head gets head_lr (highest LR)
    """
    opt_params = []
    no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}

    # --- Transformer encoder layers ---
    num_layers = model.config.num_hidden_layers  # 12 for codebert-base

    # Embeddings
    emb_params_wd = []
    emb_params_nowd = []
    for n, p in model.transformer.embeddings.named_parameters():
        if any(nd in n for nd in no_decay):
            emb_params_nowd.append(p)
        else:
            emb_params_wd.append(p)
    emb_lr = base_lr * (llrd_factor ** num_layers)
    if emb_params_wd:
        opt_params.append({"params": emb_params_wd, "lr": emb_lr,
                   "weight_decay": weight_decay})
    if emb_params_nowd:
        opt_params.append({"params": emb_params_nowd, "lr": emb_lr,
                   "weight_decay": 0.0})

    # Encoder layers
    for i in range(num_layers):
        layer = model.transformer.encoder.layer[i]
        layer_lr = base_lr * (llrd_factor ** (num_layers - i))
        wd_p, nowd_p = [], []
        for n, p in layer.named_parameters():
            if any(nd in n for nd in no_decay):
                nowd_p.append(p)
            else:
                wd_p.append(p)
        if wd_p:
            opt_params.append({"params": wd_p, "lr": layer_lr,
                               "weight_decay": weight_decay})
        if nowd_p:
            opt_params.append({"params": nowd_p, "lr": layer_lr,
                               "weight_decay": 0.0})

    # --- Classification head + feature norm ---
    head_wd, head_nowd = [], []
    for module in [model.head, model.feat_norm]:
        for n, p in module.named_parameters():
            if any(nd in n for nd in no_decay):
                head_nowd.append(p)
            else:
                head_wd.append(p)
    if head_wd:
        opt_params.append({"params": head_wd, "lr": head_lr,
                   "weight_decay": weight_decay})
    if head_nowd:
        opt_params.append({"params": head_nowd, "lr": head_lr,
                   "weight_decay": 0.0})

    return opt_params


class DeepHeadTrainer(Trainer):
    """
    Trainer subclass that injects LLRD optimizer and cosine scheduler.
    """

    def __init__(self, *args, llrd_factor=0.95, head_lr=1e-3, **kwargs):
        self.llrd_factor = llrd_factor
        self.head_lr = head_lr
        super().__init__(*args, **kwargs)

    def create_optimizer_and_scheduler(self, num_training_steps):
        """Setup LLRD optimizer and cosine schedule with warmup."""
        param_groups = get_layer_wise_lr_groups(
            self.model,
            base_lr=self.args.learning_rate,
            head_lr=self.head_lr,
            weight_decay=self.args.weight_decay,
            llrd_factor=self.llrd_factor,
        )
        self.optimizer = torch.optim.AdamW(param_groups)

        warmup_steps = int(num_training_steps * self.args.warmup_ratio)
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
        return self.optimizer, self.lr_scheduler
