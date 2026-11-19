"""Step 4 (BEAM: Data): batasi jumlah review per game, stratified berdasarkan review_score di tiap game."""
from __future__ import annotations

import polars as pl

from src.phase1.utils.io import base_parser, config_from_args, read_parquet, stage_path, write_parquet
from src.phase1.utils.logging import get_logger, log_rows
from src.phase1.utils.seed import set_seed

log = get_logger("sample_reviews")


def sample_reviews(reviews: pl.DataFrame, context_app_ids: list[int], scfg: dict, seed: int) -> pl.DataFrame:
    """Pertahankan game yang punya konteks dan minimal `min_reviews_per_game` review; ambil maksimal
    `max_reviews_per_game` per game, dengan tiap grup review_score terwakili secara proporsional."""
    cap, min_n = scfg["max_reviews_per_game"], scfg["min_reviews_per_game"]
    df = reviews.filter(pl.col("app_id").is_in(context_app_ids))
    df = df.filter(pl.len().over("app_id") >= min_n)
    df = df.sample(fraction=1.0, shuffle=True, seed=seed)  # di-shuffle sekali; semua head/cut sesudahnya jadi acak
    rank_in_group = pl.int_range(pl.len()).over(["app_id", "review_score"])
    quota = (cap * pl.len().over(["app_id", "review_score"]) / pl.len().over("app_id")).ceil()
    df = df.filter(rank_in_group < quota)
    df = df.filter(pl.int_range(pl.len()).over("app_id") < cap)
    return df.sort(["app_id", "review_id"])


def main() -> None:
    cfg = config_from_args(base_parser("Step 4: sampling review per game").parse_args())
    set_seed(cfg["seed"])
    reviews = read_parquet(stage_path(cfg, "interim", "reviews_filtered.parquet"))
    context = read_parquet(stage_path(cfg, "interim", "game_context.parquet"))
    out = sample_reviews(reviews, context["app_id"].to_list(), cfg["sampling"], cfg["seed"])
    log_rows(log, "sample_reviews", reviews.height, out.height)
    log.info("%d games kept", out["app_id"].n_unique())
    write_parquet(out, stage_path(cfg, "interim", "reviews_sampled.parquet"))


if __name__ == "__main__":
    main()
