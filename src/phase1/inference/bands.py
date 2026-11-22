"""Score bands shared by batch scoring, the standalone predictor and the API.

  off_topic   score <  threshold - margin   (memes, junk and reviews about something else; rule rejects get 0.0)
  borderline  threshold - margin <= score < threshold + margin   (mostly generic praise/complaints: annotators check these)
  relevant    score >= threshold + margin

`is_relevant` is still `score >= threshold`, so part of the borderline band is kept as relevant.
"""
from __future__ import annotations

import polars as pl

BANDS = ("off_topic", "borderline", "relevant")
DEFAULT_MARGIN = 0.15


def band_of(score: float | None, threshold: float, margin: float = DEFAULT_MARGIN) -> str:
    if score is None or score < threshold - margin:
        return "off_topic"
    return "borderline" if score < threshold + margin else "relevant"


def band_expr(threshold: float, margin: float = DEFAULT_MARGIN, score_col: str = "relevance_score") -> pl.Expr:
    s = pl.col(score_col)
    return (pl.when(s.is_null() | (s < threshold - margin)).then(pl.lit("off_topic"))
            .when(s < threshold + margin).then(pl.lit("borderline"))
            .otherwise(pl.lit("relevant")).alias("score_band"))


def margin_from(cfg: dict | None) -> float:
    return float(((cfg or {}).get("inference") or {}).get("borderline_margin", DEFAULT_MARGIN))
