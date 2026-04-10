"""Streaming test-set prediction and CSV export."""

from itertools import chain

import torch
from datasets import load_dataset
from tqdm import tqdm

from calibration import predict_with_thresholds


@torch.no_grad()
def predict_with_trainer(
    trainer_obj,
    parquet_path: str,
    output_path: str,
    max_length: int = 512,
    batch_size: int = 16,
    device: str = None,
    thresholds=None,
) -> None:
    """Run inference and write a submission file with ID,prediction."""
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

    id_col = "ID" if "ID" in first else "id" if "id" in first else None
    if "code" not in first:
        raise ValueError("Parquet must contain a 'code' column")

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
        global_idx = 0

        for batch in tqdm(batcher(stream, batch_size), desc="Predicting"):
            codes = [row["code"] for row in batch]
            if id_col:
                ids = [row[id_col] for row in batch]
            else:
                ids = list(range(global_idx, global_idx + len(batch)))
                global_idx += len(batch)

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
            logits_np = logits.cpu().numpy()

            if thresholds is not None:
                pred_labels = predict_with_thresholds(logits_np, thresholds).tolist()
            else:
                pred_labels = logits.argmax(dim=-1).cpu().tolist()

            for ex_id, pred in zip(ids, pred_labels):
                f.write(f"{ex_id},{pred}\n")

    print(f"Predictions saved to {output_path}")
