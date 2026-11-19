"""Step 5 (BEAM: Data): split berdasarkan app_id supaya satu game cuma muncul di satu split.

mode `group`: GroupShuffleSplit pada app_id (default 80/10/10).
mode `genre_holdout`: semua game yang primary genre-nya (genre pertama di list) ada di `holdout_genres` masuk ke test;
sisanya di-split train/val berdasarkan app_id (eksperimen domain-mismatch, configs/phase1_genre_holdout.yaml).
"""
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.model_selection import GroupShuffleSplit

from src.phase1.utils.io import base_parser, config_from_args, read_parquet, stage_path, write_parquet
from src.phase1.utils.logging import get_logger, log_rows
from src.phase1.utils.seed import set_seed

log = get_logger("split")

SPLITS = ("train", "val", "test")


def _group_split(app_ids: np.ndarray, ratios: list[float], seed: int) -> dict[int, str]:
    """Split app_id unik jadi train/val/test dengan dua kali GroupShuffleSplit."""
    train_r, val_r, test_r = ratios
    assignment: dict[int, str] = {}
    rest = app_ids
    if test_r > 0 and len(rest) > 1:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_r, random_state=seed)
        keep_idx, test_idx = next(gss.split(rest, groups=rest))
        assignment.update({int(a): "test" for a in rest[test_idx]})
        rest = rest[keep_idx]
    if val_r > 0 and len(rest) > 1:
        gss = GroupShuffleSplit(n_splits=1, test_size=val_r / (train_r + val_r), random_state=seed)
        train_idx, val_idx = next(gss.split(rest, groups=rest))
        assignment.update({int(a): "val" for a in rest[val_idx]})
        rest = rest[train_idx]
    assignment.update({int(a): "train" for a in rest})
    return assignment


def assign_splits(app_ids: list[int], context: pl.DataFrame, scfg: dict, seed: int) -> dict[int, str]:
    unique = np.array(sorted(set(app_ids)))
    if scfg["mode"] == "group":
        return _group_split(unique, scfg["ratios"], seed)
    if scfg["mode"] == "genre_holdout":
        holdout = {g.lower() for g in scfg["holdout_genres"]}
        if not holdout:
            raise ValueError("split.mode is genre_holdout but split.holdout_genres is empty (open decision: pick the genres)")
        primary = {r["app_id"]: r["genres"][0].lower() for r in context.select("app_id", "genres").iter_rows(named=True)
                   if r["genres"]}
        test_ids = {a for a in unique.tolist() if primary.get(a) in holdout}
        rest = np.array([a for a in unique.tolist() if a not in test_ids])
        ratios = [scfg["ratios"][0], scfg["ratios"][1], 0.0]
        assignment = _group_split(rest, ratios, seed)
        assignment.update({int(a): "test" for a in test_ids})
        return assignment
    raise ValueError(f"unknown split.mode {scfg['mode']!r}")


def split_reviews(sampled: pl.DataFrame, context: pl.DataFrame, scfg: dict, seed: int) -> dict[str, pl.DataFrame]:
    assignment = assign_splits(sampled["app_id"].unique().to_list(), context, scfg, seed)
    df = sampled.with_columns(pl.col("app_id").replace_strict(assignment, return_dtype=pl.String).alias("split"))
    return {name: df.filter(pl.col("split") == name) for name in SPLITS}


def main() -> None:
    cfg = config_from_args(base_parser("Step 5: split berdasarkan game").parse_args())
    set_seed(cfg["seed"])
    sampled = read_parquet(stage_path(cfg, "interim", "reviews_sampled.parquet"))
    context = read_parquet(stage_path(cfg, "interim", "game_context.parquet"))
    parts = split_reviews(sampled, context, cfg["split"], cfg["seed"])
    for name, part in parts.items():
        write_parquet(part, stage_path(cfg, "processed", f"split_{name}.parquet"))
        log_rows(log, f"split_{name}", sampled.height, part.height)
        log.info("split_%s: %d games", name, part["app_id"].n_unique())


if __name__ == "__main__":
    main()
