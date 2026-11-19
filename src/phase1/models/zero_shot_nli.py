"""Run `zero_shot_nli`: facebook/bart-large-mnli, premise = review, hypothesis = template with the game context.
Score = P(entailment) from the 3-way softmax. No fine-tuning."""
from __future__ import annotations

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.phase1.models.evaluate import evaluate_run
from src.phase1.models.hf_utils import predict_logits
from src.phase1.utils.io import base_parser, config_from_args, read_parquet, run_dir, stage_path
from src.phase1.utils.logging import get_logger
from src.phase1.utils.seed import set_seed

log = get_logger("zero_shot_nli")

RUN_NAME = "zero_shot_nli"


class ZeroShotScorer:
    def __init__(self, zcfg: dict):
        self.cfg = zcfg
        self.tokenizer = AutoTokenizer.from_pretrained(zcfg["model_name"])
        self.model = AutoModelForSequenceClassification.from_pretrained(zcfg["model_name"])
        label2id = {k.lower(): v for k, v in self.model.config.label2id.items()}
        if "entailment" not in label2id:
            raise KeyError(f"{zcfg['model_name']} has no 'entailment' label: {self.model.config.label2id}")
        self.entail_id = label2id["entailment"]

    def score(self, reviews: list[str], contexts: list[str]) -> np.ndarray:
        hypotheses = [self.cfg["hypothesis_template"].format(context_text=c) for c in contexts]
        logits = predict_logits(self.model, self.tokenizer, reviews, hypotheses, self.cfg["max_length"],
                                self.cfg["batch_size"], self.cfg["fp16"], desc="zero-shot")
        return torch.softmax(torch.from_numpy(logits), dim=-1)[:, self.entail_id].numpy()


def run(cfg: dict, val_pairs: pl.DataFrame, gold: pl.DataFrame, max_val_pairs: int | None = 2000) -> dict:
    """The model is never trained, so val is only used for the threshold; cap it to keep the GPU time small."""
    set_seed(cfg["seed"])
    if max_val_pairs and val_pairs.height > max_val_pairs:
        val_pairs = val_pairs.sample(n=max_val_pairs, seed=cfg["seed"])
    out = run_dir(cfg, RUN_NAME)
    scorer = ZeroShotScorer(cfg["models"]["zero_shot"])
    return evaluate_run(out, "zero_shot", scorer, val_pairs, gold, cfg, extra_info={"zero_shot": cfg["models"]["zero_shot"]})


def main() -> None:
    cfg = config_from_args(base_parser("Run zero_shot_nli").parse_args())
    val = read_parquet(stage_path(cfg, "processed", "pairs_val.parquet"))
    gold = read_parquet(stage_path(cfg, "processed", "gold_test.parquet"))
    run(cfg, val, gold)


if __name__ == "__main__":
    main()
