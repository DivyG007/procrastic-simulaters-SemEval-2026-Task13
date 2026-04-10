"""Streaming test inference for Task C baseline."""

from itertools import chain

import torch
from datasets import load_dataset
from tqdm import tqdm


@torch.no_grad()
def predict_with_trainer(trainer_obj, parquet_path: str, output_path: str, max_length: int = 256, batch_size: int = 32, device: str = None):
    """Run streaming inference and write `ID,prediction` CSV output."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = trainer_obj.model
    tokenizer = trainer_obj.tokenizer
    if tokenizer is None:
        raise ValueError("trainer_obj must have a tokenizer.")

    model.to(device)
    model.eval()

    ds = load_dataset("parquet", data_files=parquet_path, split="train", streaming=True)
    it = iter(ds)
    first = next(it)

    if not {"ID", "code"}.issubset(first.keys()):
        raise ValueError("Parquet file must contain 'ID' and 'code' columns")

    stream = chain([first], it)

    def batcher(iterator, bs):
        buf = []
        for ex in iterator:
            buf.append(ex)
            if len(buf) == bs:
                yield buf
                buf = []
        if buf:
            yield buf

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("ID,prediction\n")

        for batch in tqdm(batcher(stream, batch_size), desc="Predicting"):
            codes = [row["code"] for row in batch]
            ids = [row["ID"] for row in batch]

            enc = tokenizer(
                codes,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            pred_labels = logits.argmax(dim=-1).cpu().tolist()

            for ex_id, pred in zip(ids, pred_labels):
                f.write(f"{ex_id},{pred}\n")

    print(f"Predictions saved to {output_path}")
