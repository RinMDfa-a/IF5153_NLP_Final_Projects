"""Step 10: ubah data/output/relevant_reviews.parquet (dari score_reviews) jadi file-file yang dipakai orang lain.

    python -m src.phase1.inference.export_outputs --config configs/phase1.yaml [--top-k 30] [--max-games N]

Nulis ke `paths.export_dir` (data/output/):
  relevant_reviews.csv    cuma baris is_relevant: review_id, app_id, app_name, review_text, relevance_score, score_band.
                          Bisa langsung jadi input Phase 2 (ABSAPipeline baca `app_name` + `review_text` dari CSV).
  annotation_check.txt    SETELAH model jalan: sampel stratified dari review asli yang sudah di-skor buat dicek annotator,
                          N baris per score_band + N review yang ditolak quality rule, dengan konteks game di sebelah
                          review dan kolom `annotator_label` / `notes` yang kosong. Dipisah '|', UTF-8 dengan BOM (utils.io.write_annotation_file).
  phase2_input.json       {"1": {"app_id", "app_name", "reviews": [...], "relevance_scores": [...]}, "2": ...}
                          top-k review relevan per game, skor tertinggi duluan; bentuk keyed-nya sama dengan JSON Phase 2/3.
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from src.phase1.inference.bands import BANDS
from src.phase1.utils.io import (base_parser, config_from_args, read_parquet, resolve, stage_path,
                                 write_annotation_file)
from src.phase1.utils.logging import get_logger

log = get_logger("export_outputs")

PHASE2_CSV_COLUMNS = ["review_id", "app_id", "app_name", "review_text", "relevance_score", "score_band"]
ANNOTATION_COLUMNS = ["review_id", "app_id", "app_name", "score_band", "relevance_score", "rejected_by", "review_text",
                      "context_text", "annotator_label", "notes"]


def relevant_table(scored: pl.DataFrame) -> pl.DataFrame:
    return (scored.filter(pl.col("is_relevant"))
            .sort(["app_id", "relevance_score"], descending=[False, True])
            .select(PHASE2_CSV_COLUMNS))


def annotation_sample(scored: pl.DataFrame, context: pl.DataFrame | None, per_band: int, seed: int) -> pl.DataFrame:
    """`per_band` baris yang di-skor model dari tiap band, ditambah `per_band` reject quality rule (`rejected_by` terisi)."""
    model_scored = scored.filter(pl.col("rejected_by").is_null())
    parts = [model_scored.filter(pl.col("score_band") == b) for b in BANDS]
    parts.append(scored.filter(pl.col("rejected_by").is_not_null()))
    parts = [p.sample(n=min(per_band, p.height), seed=seed) for p in parts if p.height]
    sample = pl.concat(parts, how="vertical_relaxed") if parts else scored.clear()
    if context is not None:
        sample = sample.join(context.select("app_id", "context_text"), on="app_id", how="left")
    else:
        sample = sample.with_columns(pl.lit(None, pl.String).alias("context_text"))
    order = {b: i for i, b in enumerate(BANDS)}
    return (sample.with_columns(pl.lit("").alias("annotator_label"), pl.lit("").alias("notes"),
                                pl.col("score_band").replace_strict(order, default=99).alias("_order"),
                                pl.col("rejected_by").is_not_null().alias("_rejected"))
            .sort(["_rejected", "_order", "relevance_score"]).select(ANNOTATION_COLUMNS))


def phase2_json(scored: pl.DataFrame, top_k: int, min_reviews: int, max_games: int | None = None) -> dict:
    """Di-key per game seperti file JSON Phase 2/3; game dengan review relevan terbanyak duluan."""
    per_game = (scored.filter(pl.col("is_relevant"))
                .sort("relevance_score", descending=True)
                .group_by("app_id", maintain_order=True)
                .agg(pl.col("app_name").first(), pl.len().alias("n_relevant"),
                     pl.col("review_text").head(top_k).alias("reviews"),
                     pl.col("relevance_score").head(top_k).alias("relevance_scores"))
                .filter(pl.col("reviews").list.len() >= min_reviews)
                .sort(["n_relevant", "app_id"], descending=[True, False]))
    if max_games:
        per_game = per_game.head(max_games)
    return {str(i): {"app_id": row["app_id"], "app_name": row["app_name"], "reviews": row["reviews"],
                     "relevance_scores": [round(float(s), 4) for s in row["relevance_scores"]]}
            for i, row in enumerate(per_game.iter_rows(named=True), start=1)}


def export_outputs(scored: pl.DataFrame, context: pl.DataFrame | None, out_dir: Path, ecfg: dict, seed: int,
                   top_k: int | None = None, max_games: int | None = None) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: out_dir / name for name in ("relevant_reviews.csv", "annotation_check.txt", "phase2_input.json")}

    relevant = relevant_table(scored)
    relevant.write_csv(paths["relevant_reviews.csv"])
    log.info("relevant_reviews.csv: %d rows, %d games", relevant.height, relevant["app_id"].n_unique())

    sample = annotation_sample(scored, context, ecfg.get("annotation_per_band", 100), seed)
    write_annotation_file(sample, paths["annotation_check.txt"])
    log.info("annotation_check.txt: %d rows %s", sample.height, dict(sample["score_band"].value_counts().iter_rows()))

    games = phase2_json(scored, top_k or ecfg.get("top_k_per_game", 30), ecfg.get("min_reviews_per_game", 3), max_games)
    paths["phase2_input.json"].write_text(json.dumps(games, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("phase2_input.json: %d games", len(games))
    return paths


def main() -> None:
    parser = base_parser("Step 10: tulis deliverable CSV / JSON dari relevant_reviews.parquet")
    parser.add_argument("--top-k", type=int, default=None, help="reviews per game in phase2_input.json")
    parser.add_argument("--max-games", type=int, default=None, help="cap the number of games in phase2_input.json")
    args = parser.parse_args()
    cfg = config_from_args(args)
    scored = read_parquet(stage_path(cfg, "export", "relevant_reviews.parquet"))
    ctx_path = resolve(cfg["paths"]["game_context_file"])
    context = pl.read_parquet(ctx_path) if ctx_path.exists() else None
    paths = export_outputs(scored, context, resolve(cfg["paths"]["export_dir"]), cfg.get("export", {}), cfg["seed"],
                           args.top_k, args.max_games)
    for p in paths.values():
        log.info("wrote %s", p)


if __name__ == "__main__":
    main()
