"""Step 9 (BEAM: Inference, "Relevance Filtering"): kasih skor tiap review terhadap game-nya SENDIRI.

Output contract buat Task 2 dan 3 (jangan ganti nama kolom tanpa bilang ke pemilik task-nya):
  data/output/relevant_reviews.parquet:
    review_id, app_id, app_name, review_text, relevance_score, is_relevant, score_band, rejected_by, model_run

Review yang dibuang quality rule (`inference.include_rejected_rules`) tetap ditambahkan dengan relevance_score 0.0 dan
`rejected_by` terisi, jadi meme / ASCII art muncul di bagian bawah rentang skor, bukan hilang begitu saja.
`score_band` adalah off_topic | borderline | relevant (lihat src/phase1/inference/bands.py).
Habis itu jalankan `python -m src.phase1.inference.export_outputs` buat bikin file CSV / JSON-nya.
"""
from __future__ import annotations

import json

import numpy as np
import polars as pl

from src.phase1.inference.bands import band_expr, margin_from
from src.phase1.utils.io import base_parser, config_from_args, read_parquet, run_dir, stage_path, write_parquet
from src.phase1.utils.logging import get_logger, log_rows
from src.phase1.utils.seed import set_seed

log = get_logger("score_reviews")

OUTPUT_COLUMNS = ["review_id", "app_id", "app_name", "review_text", "relevance_score", "is_relevant", "score_band",
                  "rejected_by", "model_run"]


def load_scorer(cfg: dict, run_name: str):
    """Bangun ulang scorer yang disimpan oleh sebuah run, dari models/phase1/<run_name>/model_info.json. Return (scorer, threshold)."""
    out = run_dir(cfg, run_name)
    info_path = out / "model_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"{info_path} not found; run the {run_name} experiment first")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    kind = info["type"]
    if kind == "lexical":
        from src.phase1.models.lexical_baseline import LexicalScorer
        return LexicalScorer.load(out), info["threshold"]
    if kind == "zero_shot":
        from src.phase1.models.zero_shot_nli import ZeroShotScorer
        return ZeroShotScorer(cfg["models"]["zero_shot"]), info["threshold"]
    if kind == "cross_encoder":
        from src.phase1.models.train_cross_encoder import CrossEncoderScorer
        return CrossEncoderScorer(out / info["model_dir"], cfg["models"]["cross_encoder"], info["label_scheme"]), info["threshold"]
    raise ValueError(f"unknown model type {kind!r} in {info_path}")


def score_reviews(reviews: pl.DataFrame, context: pl.DataFrame, scorer, threshold: float, model_run: str,
                  batch_rows: int = 20_000, rejected: pl.DataFrame | None = None,
                  margin: float = 0.15) -> pl.DataFrame:
    """Kasih skor `reviews` terhadap game-nya sendiri. `rejected` (opsional, dengan kolom `rejected_by`) ditambahkan dengan
    skor 0.0 tanpa memanggil model."""
    name = pl.col("name") if "name" in context.columns else pl.lit(None, pl.String)
    ctx = context.select("app_id", "context_text", name.alias("_ctx_name"))
    df = reviews.join(ctx, on="app_id", how="inner")
    scores = []
    for start in range(0, df.height, batch_rows):  # dipotong per chunk supaya kalau crash nggak kehilangan semuanya di run yang berat memori
        chunk = df.slice(start, batch_rows)
        scores.append(np.asarray(scorer.score(chunk["review_text"].to_list(), chunk["context_text"].to_list())))
    score = np.concatenate(scores) if scores else np.zeros(0)
    df = df.with_columns(pl.Series("relevance_score", score.astype(np.float32)),
                         pl.lit(None, pl.String).alias("rejected_by"))
    if rejected is not None and rejected.height:
        rej = rejected.join(ctx, on="app_id", how="inner").with_columns(pl.lit(0.0, pl.Float32).alias("relevance_score"))
        df = pl.concat([df, rej.select(df.columns)], how="vertical_relaxed")
    if "app_name" not in df.columns:
        df = df.with_columns(pl.lit(None, pl.String).alias("app_name"))
    return df.with_columns(
        pl.coalesce("app_name", "_ctx_name").alias("app_name"),
        (pl.col("relevance_score") >= threshold).alias("is_relevant"),
        band_expr(threshold, margin),
        pl.lit(model_run).alias("model_run"),
    ).select(OUTPUT_COLUMNS)


def _cap_per_game(df: pl.DataFrame, cap: int | None, seed: int) -> pl.DataFrame:
    if not cap:
        return df
    return df.sample(fraction=1.0, shuffle=True, seed=seed).filter(pl.int_range(pl.len()).over("app_id") < cap)


def select_input(cfg: dict) -> pl.DataFrame:
    icfg = cfg["inference"]
    name = "reviews_filtered.parquet" if icfg["input"] == "filtered" else "reviews_sampled.parquet"
    return _cap_per_game(read_parquet(stage_path(cfg, "interim", name)), icfg.get("max_reviews_per_game"), cfg["seed"])


def select_rejected(cfg: dict, app_ids: pl.Series | None = None) -> pl.DataFrame | None:
    """Review yang ditolak quality rule untuk dilaporkan dengan skor 0.0 (`inference.include_rejected_rules`; kosong atau tidak ada = tidak ada)."""
    icfg = cfg["inference"]
    rules = icfg.get("include_rejected_rules") or []
    path = stage_path(cfg, "interim", "reviews_rejected.parquet")
    if not rules or not path.exists():
        return None
    df = read_parquet(path).rename({"rule_name": "rejected_by"}).filter(pl.col("rejected_by").is_in(rules))
    if app_ids is not None:
        df = df.filter(pl.col("app_id").is_in(app_ids.to_list()))
    return _cap_per_game(df, icfg.get("max_rejected_per_game"), cfg["seed"])


def main() -> None:
    parser = base_parser("Step 9: kasih skor review terhadap game-nya sendiri")
    parser.add_argument("--run", default=None, help="run folder under models/phase1/ (default: run_name from the config)")
    args = parser.parse_args()
    cfg = config_from_args(args)
    set_seed(cfg["seed"])
    name = args.run or cfg["run_name"]
    scorer, threshold = load_scorer(cfg, name)
    reviews = select_input(cfg)
    rejected = select_rejected(cfg, reviews["app_id"].unique())
    context = read_parquet(stage_path(cfg, "interim", "game_context.parquet"))
    out = score_reviews(reviews, context, scorer, threshold, name, rejected=rejected, margin=margin_from(cfg))
    log_rows(log, "score_reviews", reviews.height + (rejected.height if rejected is not None else 0), out.height)
    log.info("%.1f%% of reviews are relevant at threshold %.2f", 100 * out["is_relevant"].mean(), threshold)
    log.info("score bands: %s", dict(out["score_band"].value_counts().iter_rows()))
    write_parquet(out, stage_path(cfg, "export", "relevant_reviews.parquet"))


if __name__ == "__main__":
    main()
