"""Run `lexical_baseline`: TF-IDF cosine(review, context), threshold tuned on val. Is word overlap enough?"""
from __future__ import annotations

import joblib
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer

from src.phase1.models.evaluate import evaluate_run
from src.phase1.utils.io import base_parser, config_from_args, read_parquet, run_dir, stage_path
from src.phase1.utils.logging import get_logger
from src.phase1.utils.seed import set_seed

log = get_logger("lexical_baseline")

RUN_NAME = "lexical_baseline"


class LexicalScorer:
    def __init__(self, vectorizer: TfidfVectorizer):
        self.vectorizer = vectorizer

    def score(self, reviews: list[str], contexts: list[str]) -> np.ndarray:
        a, b = self.vectorizer.transform(reviews), self.vectorizer.transform(contexts)
        # TF-IDF rows are L2-normalized, so the row-wise dot product is the cosine similarity (already in [0, 1])
        return np.asarray(a.multiply(b).sum(axis=1)).ravel().clip(0, 1)

    def save(self, run_dir) -> None:
        joblib.dump(self.vectorizer, run_dir / "vectorizer.joblib")

    @classmethod
    def load(cls, run_dir) -> "LexicalScorer":
        return cls(joblib.load(run_dir / "vectorizer.joblib"))


def fit_scorer(train_pairs: pl.DataFrame, mcfg: dict) -> LexicalScorer:
    corpus = train_pairs["review_text"].unique().to_list() + train_pairs["context_text"].unique().to_list()
    vec = TfidfVectorizer(ngram_range=tuple(mcfg["ngram_range"]), min_df=mcfg["min_df"], sublinear_tf=True,
                          stop_words="english")
    vec.fit(corpus)
    return LexicalScorer(vec)


def run(cfg: dict, train_pairs: pl.DataFrame, val_pairs: pl.DataFrame, gold: pl.DataFrame) -> dict:
    set_seed(cfg["seed"])
    out = run_dir(cfg, RUN_NAME)
    scorer = fit_scorer(train_pairs, cfg["models"]["lexical"])
    scorer.save(out)
    return evaluate_run(out, "lexical", scorer, val_pairs, gold, cfg)


def main() -> None:
    cfg = config_from_args(base_parser("Run lexical_baseline").parse_args())
    train = read_parquet(stage_path(cfg, "processed", "pairs_train.parquet"))
    val = read_parquet(stage_path(cfg, "processed", "pairs_val.parquet"))
    gold = read_parquet(stage_path(cfg, "processed", "gold_test.parquet"))
    run(cfg, train, val, gold)


if __name__ == "__main__":
    main()
