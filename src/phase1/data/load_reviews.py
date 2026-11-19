"""Step 1 (BEAM: Data): load review Steam mentah ke interim/reviews.parquet.

Cuma baca kolom yang dibutuhkan, bersihin teks, dan kasih `review_id` yang stabil. Teks placeholder, duplikat, dan
bahasa ada di step 2, di mana setiap review yang ditolak dicatat rule yang kena.
"""
from __future__ import annotations

import polars as pl
from src.phase1.utils.io import base_parser, config_from_args, resolve, stage_path, write_parquet
from src.phase1.utils.logging import get_logger, log_rows
from src.phase1.utils.seed import set_seed
from src.phase1.utils.text_clean import clean_review

log = get_logger("load_reviews")
OUT_COLUMNS = ["review_id", "app_id", "app_name", "review_text", "review_score", "review_votes"]


def scan_reviews(cfg: dict) -> pl.LazyFrame:
    """Scan CSV review secara lazy, kolomnya di-rename ke nama kolom internal."""
    path = resolve(cfg["reviews"]["file"])
    if not path.exists():
        folder = path.parent
        seen = sorted(p.name for p in folder.iterdir()) if folder.exists() else "folder does not exist"
        raise FileNotFoundError(f"{path} not found (files in {folder.name}/: {seen}). "
                                f"Check reviews.file in the config. "
                                f"If the file is there, restart the notebook kernel and rerun from the top.")
    mapping = cfg["reviews"]["columns"]
    lf = pl.scan_csv(path, infer_schema_length=10_000, quote_char='"', truncate_ragged_lines=True,
                     schema_overrides={mapping["review_text"]: pl.String, mapping["app_name"]: pl.String})
    present = lf.collect_schema().names()
    missing = [src for src in mapping.values() if src not in present]
    if missing:
        raise KeyError(f"columns {missing} not in {path.name}; header is {present}. Fix reviews.columns in the config.")
    return lf.select([pl.col(src).alias(dst) for dst, src in mapping.items()])


def limit_by_games(lf: pl.LazyFrame, limit: int, seed: int) -> pl.LazyFrame:
    """Ambil game utuh (acak, seeded) sampai sekitar `limit` baris, jadi run kecil tetap dapat game yang lengkap."""
    counts = lf.group_by("app_id").len().collect().sample(fraction=1.0, shuffle=True, seed=seed)
    keep = counts.with_columns(pl.col("len").cum_sum().alias("_cum")).filter(pl.col("_cum") <= max(limit, 1))
    if keep.height == 0:
        keep = counts.head(1)
    return lf.filter(pl.col("app_id").is_in(keep["app_id"].to_list()))


def load_reviews(cfg: dict) -> pl.DataFrame:
    lf = scan_reviews(cfg).with_row_index("review_id")
    lf = lf.filter(pl.col("app_id").is_not_null())
    n_raw = lf.select(pl.len()).collect().item()
    if cfg.get("limit"):
        lf = limit_by_games(lf, cfg["limit"], cfg["seed"])
    df = lf.collect()
    df = df.with_columns(
        pl.col("app_id").cast(pl.Int64),
        pl.col("review_text").fill_null("").map_elements(clean_review, return_dtype=pl.String),
        pl.col("review_score").cast(pl.Int8),
        pl.col("review_votes").cast(pl.Int8, strict=False),
    )
    log_rows(log, "load_reviews", n_raw, df.height)
    return df.select(OUT_COLUMNS)


def main() -> None:
    cfg = config_from_args(base_parser("Step 1: load review mentah").parse_args())
    set_seed(cfg["seed"])
    df = load_reviews(cfg)
    write_parquet(df, stage_path(cfg, "interim", "reviews.parquet"))
    log.info("wrote interim/reviews.parquet (%d rows, %d games)", df.height, df["app_id"].n_unique())


if __name__ == "__main__":
    main()
