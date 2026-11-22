"""Step 6 (BEAM: Engineering, "Negative Sampling & Labeling"): bikin pasangan review-konteks.

Untuk tiap review r hasil sampling dari game g:
  positive         r + context(g)                                      binary 1 / graded 2
  hard_negative    r + context game lain yang punya >=1 genre sama      binary 0 / graded 1
  random_negative  r + context game yang nggak punya genre sama         binary 0 / graded 0
Negative cuma diambil dari game di split yang sama, jadi nggak ada pasangan yang lintas split.
Opsional: review yang ditolak quality rule ditambahkan sebagai pasangan own-game dengan label 0 yang lemah (quality_reject).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from src.phase1.data.split import SPLITS
from src.phase1.utils.io import base_parser, config_from_args, read_parquet, stage_path, write_parquet
from src.phase1.utils.logging import get_logger
from src.phase1.utils.seed import set_seed

log = get_logger("make_pairs")

PAIR_COLUMNS = ["pair_id", "review_id", "app_id", "paired_app_id", "review_text", "context_text", "pair_type", "label", "split"]

LABELS = {
    "binary": {"positive": 1, "hard_negative": 0, "random_negative": 0, "quality_reject": 0},
    "graded": {"positive": 2, "hard_negative": 1, "random_negative": 0, "quality_reject": 0},
}


def build_pairs(reviews: pl.DataFrame, context: pl.DataFrame, pcfg: dict, split_name: str, seed: int,
                rejected: pl.DataFrame | None = None) -> pl.DataFrame:
    labels = LABELS[pcfg["label_scheme"]]
    rng = np.random.default_rng(seed + SPLITS.index(split_name))
    ctx_text = dict(context.select("app_id", "context_text").iter_rows())
    genres = {a: set(g) for a, g in context.select("app_id", "genres").iter_rows()}

    games = sorted(a for a in reviews["app_id"].unique().to_list() if a in ctx_text)
    hard_pool = {a: [b for b in games if b != a and genres[a] & genres[b]] for a in games}
    rand_pool = {a: [b for b in games if b != a and not (genres[a] & genres[b])] for a in games}

    rows: list[tuple] = []
    n_fallback = 0

    def add(review_id, app_id, paired, text, ptype, k=0):
        rows.append((f"{split_name}-{review_id}-{ptype[0]}{k}-{paired}", review_id, app_id, paired, text,
                     ctx_text[paired], ptype, labels[ptype], split_name))

    for review_id, app_id, text in reviews.select("review_id", "app_id", "review_text").iter_rows():
        if app_id not in ctx_text:
            continue
        add(review_id, app_id, app_id, text, "positive")
        for k in range(pcfg["neg_per_pos"]):
            want_hard = rng.random() < pcfg["hard_ratio"]
            pool = hard_pool[app_id] if want_hard else rand_pool[app_id]
            ptype = "hard_negative" if want_hard else "random_negative"
            if not pool:  # misal semua game di split punya genre yang sama; pakai pool lain daripada dibuang
                pool, ptype = (rand_pool[app_id], "random_negative") if want_hard else (hard_pool[app_id], "hard_negative")
                n_fallback += 1
            if pool:
                add(review_id, app_id, pool[int(rng.integers(len(pool)))], text, ptype, k)

    if n_fallback:
        log.warning("%s: %d negatives drawn from the other pool (no candidates of the requested kind)", split_name, n_fallback)

    if pcfg["include_quality_reject"] and rejected is not None and split_name != "test":
        n_pos = reviews.height
        rej = rejected.filter(pl.col("rule_name").is_in(pcfg["quality_reject_rules"]) & pl.col("app_id").is_in(games))
        n_take = min(rej.height, int(pcfg["quality_reject_ratio"] * n_pos))
        if n_take:
            rej = rej.sample(n=n_take, seed=seed)
            for review_id, app_id, text in rej.select("review_id", "app_id", "review_text").iter_rows():
                add(review_id, app_id, app_id, text, "quality_reject")

    return pl.DataFrame(rows, schema=dict(zip(PAIR_COLUMNS, [pl.String, pl.Int64, pl.Int64, pl.Int64, pl.String, pl.String,
                                                              pl.String, pl.Int8, pl.String])), orient="row")


def main() -> None:
    cfg = config_from_args(base_parser("Step 6: bikin pasangan review-konteks").parse_args())
    set_seed(cfg["seed"])
    context = read_parquet(stage_path(cfg, "interim", "game_context.parquet"))
    rejected_path = stage_path(cfg, "interim", "reviews_rejected.parquet")
    rejected = pl.read_parquet(rejected_path) if rejected_path.exists() else None
    for name in SPLITS:
        reviews = read_parquet(stage_path(cfg, "processed", f"split_{name}.parquet"))
        pairs = build_pairs(reviews, context, cfg["pairs"], name, cfg["seed"], rejected)
        write_parquet(pairs, stage_path(cfg, "processed", f"pairs_{name}.parquet"))
        log.info("pairs_%s: %d reviews -> %d pairs; %s", name, reviews.height, pairs.height,
                 dict(pairs.group_by("pair_type").len().iter_rows()))


if __name__ == "__main__":
    main()
