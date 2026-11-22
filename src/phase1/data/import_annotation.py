"""Step 7b (BEAM: Actor Annotator): import annotation/labeled.txt (atau labeled.csv) ke processed/gold_test.parquet.

File label bisa berupa .txt dipisah '|' seperti hasil export, atau disimpan ulang dari Excel sebagai CSV (`,`/`;`) atau teks tab.

Semua metrik yang dilaporkan pakai `gold_label`. Kalau belum ada file label dan `annotation.allow_auto_labels` true,
auto label dipakai sebagai gantinya dan tiap baris ditandai `label_source = "auto"` biar metriknya nggak dikira gold.
"""
from __future__ import annotations

import polars as pl
from src.phase1.utils.io import (base_parser, config_from_args, read_annotation_file, read_parquet, stage_path,
                                 write_parquet)
from src.phase1.utils.logging import get_logger
log = get_logger("import_annotation")


def import_annotation(test_pairs: pl.DataFrame, labeled: pl.DataFrame | None, scheme: str, allow_auto: bool,
                      to_label: pl.DataFrame | None = None) -> pl.DataFrame:
    valid = {"binary": {0, 1}, "graded": {0, 1, 2}}[scheme]
    if labeled is None:
        if not allow_auto:
            raise FileNotFoundError("annotation/labeled.txt (or .csv) not found and annotation.allow_auto_labels is false")
        if to_label is None:
            raise FileNotFoundError("annotation/to_label.txt not found; run data.export_annotation first")
        log.warning("no labeled.txt/.csv: using AUTO labels as gold_label (label_source=auto). Metrics are NOT gold.")
        ids = to_label.select("pair_id")
        return (test_pairs.join(ids, on="pair_id", how="inner")
                .with_columns(pl.col("label").alias("gold_label"), pl.lit("auto").alias("label_source")))

    labeled = labeled.with_columns(pl.col("gold_label").cast(pl.Int64, strict=False))
    unlabeled = labeled.filter(pl.col("gold_label").is_null()).height
    if unlabeled:
        log.warning("%d rows have no gold_label and are dropped", unlabeled)
    labeled = labeled.filter(pl.col("gold_label").is_not_null())
    bad = labeled.filter(~pl.col("gold_label").is_in(sorted(valid)))
    if bad.height:
        raise ValueError(f"{bad.height} gold labels outside {sorted(valid)} for scheme {scheme!r}: {bad['pair_id'].to_list()[:5]}")
    notes = labeled.select("pair_id", "gold_label", "note")
    out = test_pairs.join(notes, on="pair_id", how="inner").with_columns(pl.lit("human").alias("label_source"))
    flipped = out.filter(pl.col("gold_label") != pl.col("label")).height
    log.info("%d gold rows, %d disagree with the auto label", out.height, flipped)
    return out


def _first_existing(cfg: dict, names: list[str]):
    for name in names:
        path = stage_path(cfg, "annotation", name)
        if path.exists():
            return path
    return None


def main() -> None:
    cfg = config_from_args(base_parser("Step 7b: import hasil anotasi").parse_args())
    pairs = read_parquet(stage_path(cfg, "processed", "pairs_test.parquet"))
    labeled_path = _first_existing(cfg, ["labeled.txt", "labeled.csv"])
    to_label_path = _first_existing(cfg, ["to_label.txt", "to_label.csv"])
    labeled = read_annotation_file(labeled_path) if labeled_path else None
    to_label = read_annotation_file(to_label_path) if to_label_path else None
    log.info("labeled file: %s", labeled_path or "none (auto labels)")
    gold = import_annotation(pairs, labeled, cfg["pairs"]["label_scheme"], cfg["annotation"]["allow_auto_labels"], to_label)
    write_parquet(gold, stage_path(cfg, "processed", "gold_test.parquet"))
    log.info("gold_test: %d rows, label_source=%s", gold.height, gold["label_source"].unique().to_list())


if __name__ == "__main__":
    main()
