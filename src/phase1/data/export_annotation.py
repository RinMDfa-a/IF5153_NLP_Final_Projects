"""Step 7a (BEAM: Actor Annotator): export sampel stratified dari TEST split buat dilabeli manusia.

Nulis annotation/to_label.txt (dipisah '|', lihat utils.io.write_annotation_file). Annotator ngisi `gold_label` dan
simpan hasilnya sebagai annotation/labeled.txt (atau labeled.csv kalau disimpan ulang dari Excel).

Annotator harus (1) cari review game sendiri yang sebenarnya off-topic (set 0), dan (2) cek hard negative yang
sebenarnya relevan (review generik yang cocok juga buat game lain; set 1 di graded scheme).
"""
from __future__ import annotations

import polars as pl

from src.phase1.utils.io import base_parser, config_from_args, read_parquet, stage_path, write_annotation_file
from src.phase1.utils.logging import get_logger
from src.phase1.utils.seed import set_seed

log = get_logger("export_annotation")

COLUMNS = ["pair_id", "review_text", "game_name", "context_text", "pair_type", "auto_label", "gold_label", "note"]


def stratified_sample(pairs: pl.DataFrame, gold_size: int, weights: dict[str, float], seed: int) -> pl.DataFrame:
    """Sample `gold_size` pasangan yang dibagi ke tiap pair type sesuai `weights`; stratum yang barisnya kurang dikasih semua yang ada."""
    parts = []
    for ptype, w in weights.items():
        stratum = pairs.filter(pl.col("pair_type") == ptype)
        n = min(stratum.height, round(gold_size * w))
        if n:
            parts.append(stratum.sample(n=n, seed=seed))
    return pl.concat(parts) if parts else pairs.head(0)


def export_annotation(test_pairs: pl.DataFrame, context: pl.DataFrame, acfg: dict, seed: int) -> pl.DataFrame:
    sample = stratified_sample(test_pairs, acfg["gold_size"], acfg["strata_weights"], seed)
    names = context.select(pl.col("app_id").alias("paired_app_id"), pl.col("name").alias("game_name"))
    out = sample.join(names, on="paired_app_id", how="left").with_columns(
        pl.col("label").alias("auto_label"),
        pl.lit(None, dtype=pl.String).alias("gold_label"),
        pl.lit("").alias("note"),
    )
    return out.select(COLUMNS).sample(fraction=1.0, shuffle=True, seed=seed)  # di-shuffle supaya annotator nggak terpengaruh urutan tipe


def main() -> None:
    cfg = config_from_args(base_parser("Step 7a: export sampel anotasi").parse_args())
    set_seed(cfg["seed"])
    pairs = read_parquet(stage_path(cfg, "processed", "pairs_test.parquet"))
    context = read_parquet(stage_path(cfg, "interim", "game_context.parquet"))
    out = export_annotation(pairs, context, cfg["annotation"], cfg["seed"])
    path = stage_path(cfg, "annotation", "to_label.txt")
    write_annotation_file(out, path)
    log.info("wrote %s (%d rows): %s", path, out.height, dict(out.group_by("pair_type").len().iter_rows()))


if __name__ == "__main__":
    main()
