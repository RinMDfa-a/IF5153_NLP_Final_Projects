"""Shared evaluation: threshold chosen on val, metrics reported on the gold test set.

Writes models/phase1/<run_name>/metrics.json, errors.csv and model_info.json (the chosen threshold, read at inference).
Every scorer exposes `score(reviews: list[str], contexts: list[str]) -> np.ndarray` with values in [0, 1].
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.phase1.utils.io import save_run_meta
from src.phase1.utils.logging import get_logger

log = get_logger("evaluate")


class Scorer(Protocol):
    def score(self, reviews: list[str], contexts: list[str]) -> np.ndarray: ...


def relevant_min_label(cfg: dict) -> int:
    """Smallest label that counts as 'relevant' for P/R/F1."""
    return 1 if cfg["pairs"]["label_scheme"] == "binary" else cfg["evaluation"]["relevant_min_label"]


def to_binary(labels: np.ndarray, min_label: int) -> np.ndarray:
    return (np.asarray(labels) >= min_label).astype(int)


def choose_threshold(scores: np.ndarray, y_true: np.ndarray, grid: int = 101) -> float:
    """Threshold on [0, 1] that maximizes F1 of the relevant class."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0, 1, grid):
        _, _, f1, _ = precision_recall_fscore_support(y_true, (scores >= t).astype(int), average="binary",
                                                      pos_label=1, zero_division=0)
        if f1 >= best_f1:  # ties go to the higher (stricter) threshold
            best_t, best_f1 = float(t), float(f1)
    return best_t


def compute_metrics(df: pl.DataFrame, threshold: float, label_col: str, min_label: int, graded: bool) -> dict:
    """df needs: score, pair_type and `label_col`."""
    scores = df["score"].to_numpy()
    y_true = to_binary(df[label_col].to_numpy(), min_label)
    y_pred = (scores >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    metrics = {"n": int(df.height), "threshold": threshold, "precision": float(p), "recall": float(r), "f1": float(f1),
               "accuracy": float(accuracy_score(y_true, y_pred))}
    if graded and df.height > 2 and len(set(df[label_col].to_list())) > 1:
        rho = spearmanr(scores, df[label_col].to_numpy()).statistic
        metrics["spearman"] = None if np.isnan(rho) else float(rho)
    by_type = {}
    for ptype in df["pair_type"].unique().to_list():
        sub = df.filter(pl.col("pair_type") == ptype)
        pred = (sub["score"].to_numpy() >= threshold).astype(int)
        true = to_binary(sub[label_col].to_numpy(), min_label)
        # recall on rows that are truly relevant, accuracy on the rest; both reduce to "share predicted correctly"
        by_type[ptype] = {"n": int(sub.height), "accuracy": float((pred == true).mean()),
                          "share_predicted_relevant": float(pred.mean())}
    metrics["by_pair_type"] = by_type
    return metrics


def score_pairs(scorer: Scorer, pairs: pl.DataFrame) -> pl.DataFrame:
    scores = scorer.score(pairs["review_text"].to_list(), pairs["context_text"].to_list())
    return pairs.with_columns(pl.Series("score", np.asarray(scores, dtype=np.float32)))


def evaluate_run(run_dir: Path, run_type: str, scorer: Scorer, val_pairs: pl.DataFrame, eval_pairs: pl.DataFrame,
                 cfg: dict, extra_info: dict | None = None) -> dict:
    """Pick the threshold on val, report on `eval_pairs` (gold test), write metrics/errors/model_info."""
    run_dir.mkdir(parents=True, exist_ok=True)
    min_label = relevant_min_label(cfg)
    graded = cfg["pairs"]["label_scheme"] == "graded"

    val = score_pairs(scorer, val_pairs)
    threshold = choose_threshold(val["score"].to_numpy(), to_binary(val["label"].to_numpy(), min_label),
                                 cfg["evaluation"]["threshold_grid"])
    label_col = "gold_label" if "gold_label" in eval_pairs.columns else "label"
    test = score_pairs(scorer, eval_pairs)
    metrics = compute_metrics(test, threshold, label_col, min_label, graded)
    metrics["run_name"] = run_dir.name
    metrics["label_source"] = (test["label_source"].unique().to_list() if "label_source" in test.columns else ["auto"])
    metrics["val_f1_at_threshold"] = compute_metrics(val, threshold, "label", min_label, graded)["f1"]

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "model_info.json").write_text(json.dumps({"type": run_type, "threshold": threshold, **(extra_info or {})},
                                                        indent=2), encoding="utf-8")
    pred = (test["score"].to_numpy() >= threshold).astype(int)
    wrong = test.filter(pl.Series(pred != to_binary(test[label_col].to_numpy(), min_label)))
    wrong = wrong.select("pair_id", "pair_type", pl.col(label_col).alias("label"), "score", "review_text", "context_text") \
                 .with_columns(pl.lit("").alias("error_tag"))  # error_tag: fill in sarcasm / hyperbole during error analysis
    wrong.write_csv(run_dir / "errors.csv")
    save_run_meta(run_dir, cfg)
    log.info("%s: P=%.3f R=%.3f F1=%.3f Acc=%.3f @thr=%.2f (%s labels)", run_dir.name, metrics["precision"],
             metrics["recall"], metrics["f1"], metrics["accuracy"], threshold, metrics["label_source"])
    return metrics
