"""Simple, standalone inference: score your own reviews against a game with a trained Task 1 model.

Needs only a model folder (`model_info.json` + `model/`, as written by training to models/phase1/<run>/) and a table with
a `review_text` column plus the game's context in ONE of these forms:
  1. `context_text`                     a ready context string
  2. `genres` + `description`           (optional `categories`); lists or ';'-separated strings; the context is built for you
  3. `app_id` + --game-context FILE     a parquet/csv with `app_id` and `context_text` (e.g. data/interim/game_context.parquet)

    python -m src.phase1.inference.predict --run models/phase1/task_adapted --input reviews.csv --output scored.parquet

Output = the input columns + `context_text`, `rejected_by` (cheap quality rule that fired, else null),
`relevance_score` in [0, 1] and `is_relevant` (score >= the threshold saved with the model).
Reviews rejected by a quality rule are not scored (`is_relevant = False`).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from src.phase1.data.quality_filter import TEXT_RULES
from src.phase1.inference.bands import band_expr, margin_from
from src.phase1.models.hf_utils import get_device, predict_logits
from src.phase1.models.train_cross_encoder import logits_to_score
from src.phase1.utils.context_text import filter_categories, make_context_text, split_semicolon, truncate_description
from src.phase1.utils.io import load_config, resolve
from src.phase1.utils.text_clean import clean_review, strip_html

# Row-level rules only: duplicates need the whole dataset and language ID is a separate dependency.
ROW_RULES = ["empty_or_placeholder", "ascii_art", "repeated_chars", "repeated_words", "too_short"]


class RelevanceModel:
    """A trained cross-encoder plus its decision threshold."""

    def __init__(self, run_dir: str | Path = "models/phase1/task_adapted", batch_size: int = 64, max_length: int = 256,
                 fp16: bool = True):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        run = resolve(run_dir)
        info_path = run / "model_info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"{info_path} not found: {run} must contain model_info.json and model/")
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if info["type"] != "cross_encoder":
            raise ValueError(f"{run.name} is a {info['type']!r} run; inference needs a cross_encoder run")
        self.threshold: float = info["threshold"]
        self.run_name: str = run.name
        self.scheme: str = info["label_scheme"]
        model_dir = run / info["model_dir"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(get_device()).eval()  # load onto the GPU once, not on the first request
        self.batch_size, self.max_length, self.fp16 = batch_size, max_length, fp16

    def score(self, reviews: list[str], contexts: list[str]) -> np.ndarray:
        logits = predict_logits(self.model, self.tokenizer, reviews, contexts, self.max_length, self.batch_size,
                                self.fp16, desc="scoring")
        return logits_to_score(logits, self.scheme)


def build_contexts(df: pl.DataFrame, game_context: pl.DataFrame | None, cfg: dict) -> list[str]:
    """One context string per row, from whichever of the three supported forms the table provides."""
    if "context_text" in df.columns:
        return df["context_text"].fill_null("").to_list()
    ccfg = cfg["context"]
    if "genres" in df.columns and "description" in df.columns:
        cats = df["categories"].to_list() if "categories" in df.columns else [None] * df.height
        return [make_context_text(split_semicolon(g), filter_categories(split_semicolon(c), ccfg["low_info_categories"]),
                                  truncate_description(strip_html(d), ccfg["max_description_chars"]))
                for g, c, d in zip(df["genres"].to_list(), cats, df["description"].to_list())]
    if "app_id" in df.columns and game_context is not None:
        lookup = dict(game_context.select("app_id", "context_text").iter_rows())
        missing = sorted({a for a in df["app_id"].to_list() if a not in lookup})
        if missing:
            raise KeyError(f"{len(missing)} app_id(s) not in the game context file, e.g. {missing[:5]}")
        return [lookup[a] for a in df["app_id"].to_list()]
    raise ValueError("need one of: a `context_text` column; `genres` + `description` columns; "
                     "or an `app_id` column together with --game-context")


def first_rule_fired(text: str, qcfg: dict) -> str | None:
    for name in ROW_RULES:
        if TEXT_RULES[name](text, qcfg):
            return name
    return None


def predict_df(df: pl.DataFrame, model, game_context: pl.DataFrame | None = None, quality_rules: bool = True,
               cfg: dict | None = None) -> pl.DataFrame:
    """Score every row of `df`. `model` needs `.score(reviews, contexts)` and `.threshold` (see RelevanceModel)."""
    if "review_text" not in df.columns:
        raise ValueError(f"input needs a `review_text` column, got {df.columns}")
    cfg = cfg or load_config("configs/phase1.yaml")
    reviews = [clean_review(t) for t in df["review_text"].fill_null("").to_list()]
    contexts = build_contexts(df, game_context, cfg)
    rejected = [first_rule_fired(t, cfg["quality"]) if quality_rules else None for t in reviews]

    keep = [i for i, r in enumerate(rejected) if r is None]
    scores = [0.0] * df.height  # junk caught by a quality rule is not sent to the model and scores 0
    if keep:
        for i, s in zip(keep, model.score([reviews[i] for i in keep], [contexts[i] for i in keep])):
            scores[i] = float(s)
    return df.with_columns(
        pl.Series("context_text", contexts, dtype=pl.String),
        pl.Series("rejected_by", rejected, dtype=pl.String),
        pl.Series("relevance_score", scores, dtype=pl.Float64),
    ).with_columns(
        (pl.col("relevance_score") >= model.threshold).alias("is_relevant"),
        band_expr(model.threshold, margin_from(cfg)),
    )


def _read_table(path: str) -> pl.DataFrame:
    p = resolve(path)
    return pl.read_parquet(p) if p.suffix == ".parquet" else pl.read_csv(p, infer_schema_length=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="models/phase1/task_adapted", help="model folder (model_info.json + model/)")
    parser.add_argument("--input", required=True, help="csv or parquet with a review_text column + game context")
    parser.add_argument("--output", default=None, help="csv or parquet (default: print a preview only)")
    parser.add_argument("--game-context", default=None, help="parquet/csv with app_id + context_text")
    parser.add_argument("--no-quality-rules", action="store_true", help="score everything, skip the cheap quality rules")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    model = RelevanceModel(args.run, batch_size=args.batch_size)
    game_context = _read_table(args.game_context) if args.game_context else None
    if game_context is not None:
        game_context = game_context.with_columns(pl.col("app_id").cast(pl.Int64))
    df = _read_table(args.input)
    if "app_id" in df.columns:
        df = df.with_columns(pl.col("app_id").cast(pl.Int64))
    out = predict_df(df, model, game_context, quality_rules=not args.no_quality_rules)
    print(f"threshold {model.threshold:.2f} | {out['is_relevant'].sum()} of {out.height} relevant, "
          f"{out['rejected_by'].is_not_null().sum()} rejected by a quality rule")
    if args.output:
        path = resolve(args.output)
        out.write_parquet(path) if path.suffix == ".parquet" else out.write_csv(path)
        print("wrote", path)
    else:
        with pl.Config(tbl_rows=30, fmt_str_lengths=70, tbl_width_chars=200):
            print(out.select("review_text", "rejected_by", "relevance_score", "score_band", "is_relevant"))


if __name__ == "__main__":
    main()
