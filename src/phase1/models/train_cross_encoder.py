"""Runs `task_adapted` (main model) and `cross_genre` (trained on the genre-holdout split).

Cross-encoder over a (review, context) pair. Binary scheme: 2 classes, score = softmax P(relevant).
Graded scheme: regression head (num_labels=1, MSE), score = prediction / 2 clipped to [0, 1].
Plain PyTorch loop with fp16 autocast and gradient accumulation so it fits an 8 GB laptop GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src.phase1.models.evaluate import choose_threshold, evaluate_run, relevant_min_label, to_binary
from src.phase1.models.hf_utils import get_device, predict_logits
from src.phase1.utils.io import base_parser, config_from_args, read_parquet, run_dir, stage_path
from src.phase1.utils.logging import get_logger
from src.phase1.utils.seed import set_seed

log = get_logger("train_cross_encoder")


class CrossEncoderScorer:
    def __init__(self, model_dir: Path, ccfg: dict, scheme: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.ccfg, self.scheme = ccfg, scheme

    def score(self, reviews: list[str], contexts: list[str]) -> np.ndarray:
        logits = predict_logits(self.model, self.tokenizer, reviews, contexts, self.ccfg["max_length"],
                                self.ccfg["eval_batch_size"], self.ccfg["fp16"], desc="cross-encoder")
        return logits_to_score(logits, self.scheme)


def logits_to_score(logits: np.ndarray, scheme: str) -> np.ndarray:
    if scheme == "binary":
        return torch.softmax(torch.from_numpy(logits), dim=-1)[:, 1].numpy()
    return np.clip(logits[:, 0] / 2.0, 0.0, 1.0)


def _collate(tokenizer, max_length):
    def fn(batch):
        enc = tokenizer([b[0] for b in batch], [b[1] for b in batch], truncation="longest_first", max_length=max_length,
                        padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([b[2] for b in batch])
        return enc
    return fn


def train(cfg: dict, train_pairs: pl.DataFrame, val_pairs: pl.DataFrame, out: Path) -> Path:
    """Fine-tune, keep the epoch with the best val F1, save it to out/model. Returns the model directory."""
    ccfg, scheme = cfg["models"]["cross_encoder"], cfg["pairs"]["label_scheme"]
    device = get_device()
    if device.type != "cuda":
        log.warning("CUDA is not available; training on CPU will be very slow")
    if ccfg.get("max_train_pairs") and train_pairs.height > ccfg["max_train_pairs"]:
        train_pairs = train_pairs.sample(n=ccfg["max_train_pairs"], seed=cfg["seed"])
    if ccfg.get("max_val_pairs") and val_pairs.height > ccfg["max_val_pairs"]:
        val_pairs = val_pairs.sample(n=ccfg["max_val_pairs"], seed=cfg["seed"])

    num_labels = 2 if scheme == "binary" else 1
    tokenizer = AutoTokenizer.from_pretrained(ccfg["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(ccfg["model_name"], num_labels=num_labels).to(device)

    label_dtype = int if scheme == "binary" else float
    rows = list(zip(train_pairs["review_text"].to_list(), train_pairs["context_text"].to_list(),
                    [label_dtype(x) for x in train_pairs["label"].to_list()]))
    gen = torch.Generator().manual_seed(cfg["seed"])
    loader = DataLoader(rows, batch_size=ccfg["batch_size"], shuffle=True, generator=gen,
                        collate_fn=_collate(tokenizer, ccfg["max_length"]))

    steps_per_epoch = max(len(loader) // ccfg["grad_accum_steps"], 1)
    total_steps = steps_per_epoch * ccfg["epochs"]
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    params = [{"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
               "weight_decay": ccfg["weight_decay"]},
              {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0}]
    optimizer = torch.optim.AdamW(params, lr=ccfg["lr"])
    scheduler = get_linear_schedule_with_warmup(optimizer, int(ccfg["warmup_ratio"] * total_steps), total_steps)
    use_amp = ccfg["fp16"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = torch.nn.CrossEntropyLoss() if scheme == "binary" else torch.nn.MSELoss()

    min_label = relevant_min_label(cfg)
    val_y = to_binary(val_pairs["label"].to_numpy(), min_label)
    best_f1, history = -1.0, []
    model_dir = out / "model"

    for epoch in range(ccfg["epochs"]):
        model.train()
        running, n_seen = 0.0, 0
        optimizer.zero_grad()
        for step, batch in enumerate(tqdm(loader, desc=f"epoch {epoch + 1}/{ccfg['epochs']}"), start=1):
            labels = batch.pop("labels").to(device)
            batch = batch.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(**batch).logits
            logits = logits.float() if scheme == "binary" else logits.float().squeeze(-1)
            loss = loss_fn(logits, labels) / ccfg["grad_accum_steps"]
            scaler.scale(loss).backward()
            running += loss.item() * ccfg["grad_accum_steps"]
            n_seen += 1
            if step % ccfg["grad_accum_steps"] == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

        logits = predict_logits(model, tokenizer, val_pairs["review_text"].to_list(), val_pairs["context_text"].to_list(),
                                ccfg["max_length"], ccfg["eval_batch_size"], ccfg["fp16"], desc="val")
        scores = logits_to_score(logits, scheme)
        thr = choose_threshold(scores, val_y, cfg["evaluation"]["threshold_grid"])
        pred = (scores >= thr).astype(int)
        tp = int(((pred == 1) & (val_y == 1)).sum())
        f1 = 2 * tp / max(int(pred.sum()) + int(val_y.sum()), 1)
        history.append({"epoch": epoch + 1, "train_loss": running / max(n_seen, 1), "val_f1": f1, "val_threshold": thr})
        log.info("epoch %d: train_loss=%.4f val_f1=%.4f thr=%.2f", epoch + 1, history[-1]["train_loss"], f1, thr)
        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)

    (out / "train_log.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    del model, optimizer
    torch.cuda.empty_cache()
    return model_dir


def run(cfg: dict, train_pairs: pl.DataFrame, val_pairs: pl.DataFrame, gold: pl.DataFrame, run_name: str | None = None) -> dict:
    set_seed(cfg["seed"])
    name = run_name or cfg["run_name"]
    out = run_dir(cfg, name)
    model_dir = train(cfg, train_pairs, val_pairs, out)
    ccfg = cfg["models"]["cross_encoder"]
    scorer = CrossEncoderScorer(model_dir, ccfg, cfg["pairs"]["label_scheme"])
    eval_val = val_pairs.sample(n=min(val_pairs.height, ccfg.get("max_val_pairs") or val_pairs.height), seed=cfg["seed"])
    return evaluate_run(out, "cross_encoder", scorer, eval_val, gold, cfg,
                        extra_info={"model_dir": "model", "label_scheme": cfg["pairs"]["label_scheme"],
                                    "base_model": ccfg["model_name"]})


def main() -> None:
    cfg = config_from_args(base_parser("Train a cross-encoder (task_adapted / cross_genre)").parse_args())
    train_pairs = read_parquet(stage_path(cfg, "processed", "pairs_train.parquet"))
    val_pairs = read_parquet(stage_path(cfg, "processed", "pairs_val.parquet"))
    gold = read_parquet(stage_path(cfg, "processed", "gold_test.parquet"))
    run(cfg, train_pairs, val_pairs, gold)


if __name__ == "__main__":
    main()
