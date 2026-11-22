"""Batched GPU inference helper shared by the zero-shot NLI and cross-encoder scorers."""
from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def predict_logits(model, tokenizer, texts_a: list[str], texts_b: list[str], max_length: int, batch_size: int,
                   fp16: bool = True, desc: str | None = None) -> np.ndarray:
    """Logits for sentence pairs, shape [N, num_labels], in the input order.

    Pairs are processed in length order (less padding). `longest_first` truncation trims the longer text, which
    in practice is the review because the game context is short; it also never fails on very long reviews.
    """
    device = get_device()
    model.to(device).eval()
    order = np.argsort([len(a) + len(b) for a, b in zip(texts_a, texts_b)])
    out: list[np.ndarray | None] = [None] * len(order)
    use_amp = fp16 and device.type == "cuda"
    batches = range(0, len(order), batch_size)
    for start in tqdm(batches, desc=desc, disable=desc is None or len(order) < 2 * batch_size):
        idx = order[start:start + batch_size]
        enc = tokenizer([texts_a[i] for i in idx], [texts_b[i] for i in idx], truncation="longest_first",
                        max_length=max_length, padding=True, return_tensors="pt").to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(**enc).logits.float().cpu().numpy()
        for row, i in zip(logits, idx):
            out[i] = row
    return np.stack(out) if out else np.zeros((0, model.config.num_labels))
