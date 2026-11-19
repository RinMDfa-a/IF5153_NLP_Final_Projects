"""Phase 1 HTTP API: relevance scoring / filtering as JSON, for the Phase 3 demo CLI (exposed with ngrok).

    uvicorn src.phase1.api.server:app --port 8000        # from the repository root
    ngrok http 8000                                      # then PHASE_1_BASE_ENDPOINT=<ngrok url> in the CLI's .env

Environment variables:
  PHASE1_MODEL_DIR   model folder with model_info.json + model/      (default models/phase1/task_adapted)
  PHASE1_CONFIG      config file                                       (default configs/phase1.yaml)
  PHASE1_API_KEY     if set, every /phase1/* request needs header `X-API-Key: <value>` (the ngrok URL is public)
  PHASE1_STEAM_API   "0" disables the Steam Store API fallback for app_ids missing from game_context.parquet

Endpoints (request/response examples in docs/phase1_api.md):
  GET  /health          model, threshold, device, number of known games
  POST /phase1/score    {"reviews": [{"review_text", "app_id" | "context_text" | "genres"+"description"}]}
  POST /phase1/filter   {"1": {"app_id", "app_name"?, "reviews": [str, ...]}, "2": ...}   (same keyed shape as Phase 2/3)
"""
from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from typing import Any

import polars as pl
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from src.phase1.inference.predict import predict_df
from src.phase1.utils.context_text import filter_categories, make_context_text, split_semicolon, truncate_description
from src.phase1.utils.io import load_config, resolve
from src.phase1.utils.text_clean import strip_html

MAX_REVIEWS_PER_REQUEST = 5000


# ---------- request / response schemas ----------

class ReviewIn(BaseModel):
    review_text: str
    app_id: int | None = None
    context_text: str | None = None
    genres: str | list[str] | None = None
    categories: str | list[str] | None = None
    description: str | None = None


class ScoreRequest(BaseModel):
    reviews: list[ReviewIn] = Field(..., min_length=1, max_length=MAX_REVIEWS_PER_REQUEST)


class ScoredReview(BaseModel):
    review_text: str
    app_id: int | None
    relevance_score: float
    is_relevant: bool
    score_band: str
    rejected_by: str | None


class ScoreResponse(BaseModel):
    model_run: str
    threshold: float
    results: list[ScoredReview]


class GameIn(BaseModel):
    """One game in the keyed Phase 2/3 format. Unknown extra keys are passed through unchanged."""
    model_config = ConfigDict(extra="allow")
    reviews: list[str] = Field(..., min_length=1)
    app_id: int | None = None
    app_name: str | None = None
    context_text: str | None = None
    genres: str | list[str] | None = None
    categories: str | list[str] | None = None
    description: str | None = None


# ---------- game context ----------

class ContextStore:
    """app_id -> (name, context_text), from game_context.parquet plus (optionally) the Steam Store API."""

    def __init__(self, game_context: pl.DataFrame | None, cfg: dict, use_steam_api: bool):
        self.cfg, self.use_steam_api = cfg, use_steam_api
        self.games: dict[int, tuple[str | None, str]] = {}
        if game_context is not None:
            names = game_context["name"] if "name" in game_context.columns else [None] * game_context.height
            for app_id, name, ctx in zip(game_context["app_id"], names, game_context["context_text"]):
                self.games[int(app_id)] = (name, ctx)
        self._lock = threading.Lock()

    def from_fields(self, genres, categories, description) -> str | None:
        if genres is None or description is None:
            return None
        ccfg = self.cfg["context"]
        return make_context_text(split_semicolon(genres),
                                 filter_categories(split_semicolon(categories), ccfg["low_info_categories"]),
                                 truncate_description(strip_html(description), ccfg["max_description_chars"]))

    def lookup(self, app_id: int) -> tuple[str | None, str]:
        if app_id in self.games:
            return self.games[app_id]
        if self.use_steam_api:
            with self._lock:
                from src.phase1.data.steam_api import fetch_missing
                row = fetch_missing([app_id], self.cfg).get(app_id)
            if row and row.get("success") and row.get("genres") and row.get("short_description"):
                ctx = self.from_fields(row["genres"], row["categories"], row["short_description"])
                self.games[app_id] = (row.get("name") or None, ctx)
                return self.games[app_id]
        raise HTTPException(422, f"app_id {app_id} is not in the game context and the Steam Store API has no "
                                 f"genres/description for it; send `context_text` or `genres` + `description` instead")

    def resolve(self, item: ReviewIn | GameIn, where: str) -> tuple[str | None, str]:
        if item.context_text:
            return None, item.context_text
        ctx = self.from_fields(item.genres, item.categories, item.description)
        if ctx:
            return None, ctx
        if item.app_id is not None:
            return self.lookup(item.app_id)
        raise HTTPException(422, f"{where}: needs `app_id`, `context_text`, or `genres` + `description`")


# ---------- app ----------

def _check_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    expected = request.app.state.api_key
    if expected and x_api_key != expected:
        raise HTTPException(401, "missing or wrong X-API-Key header")


def create_app(model=None, cfg: dict | None = None, game_context: pl.DataFrame | None = None,
               api_key: str | None = None, use_steam_api: bool | None = None) -> FastAPI:
    """Build the app. Arguments left as None are loaded from the environment at startup (tests inject fakes)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        conf = cfg or load_config(os.environ.get("PHASE1_CONFIG", "configs/phase1.yaml"))
        mdl = model
        if mdl is None:
            from src.phase1.inference.predict import RelevanceModel
            mdl = RelevanceModel(os.environ.get("PHASE1_MODEL_DIR", "models/phase1/task_adapted"))
        gc = game_context
        if gc is None:
            path = resolve(conf["paths"]["game_context_file"])
            gc = pl.read_parquet(path) if path.exists() else None
        steam = use_steam_api if use_steam_api is not None else os.environ.get("PHASE1_STEAM_API", "1") != "0"
        app.state.cfg, app.state.model = conf, mdl
        app.state.contexts = ContextStore(gc, conf, steam)
        app.state.api_key = api_key if api_key is not None else os.environ.get("PHASE1_API_KEY") or None
        app.state.score_lock = threading.Lock()  # one GPU batch at a time
        yield

    app = FastAPI(title="Phase 1: review relevance", version="1.0", lifespan=lifespan)

    def score_rows(review_texts: list[str], contexts: list[str]) -> pl.DataFrame:
        df = pl.DataFrame({"review_text": review_texts, "context_text": contexts}, schema={
            "review_text": pl.String, "context_text": pl.String})
        with app.state.score_lock:
            return predict_df(df, app.state.model, cfg=app.state.cfg)

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        m = request.app.state.model
        device = None
        if hasattr(m, "model"):
            try:
                device = str(next(m.model.parameters()).device)
            except StopIteration:
                pass
        return {"status": "ok", "model_run": getattr(m, "run_name", None), "threshold": m.threshold,
                "device": device, "known_games": len(request.app.state.contexts.games),
                "auth": bool(request.app.state.api_key)}

    @app.post("/phase1/score", response_model=ScoreResponse, dependencies=[Depends(_check_key)])
    def score(req: ScoreRequest, request: Request) -> ScoreResponse:
        store: ContextStore = request.app.state.contexts
        contexts = [store.resolve(r, f"reviews[{i}]")[1] for i, r in enumerate(req.reviews)]
        out = score_rows([r.review_text for r in req.reviews], contexts)
        results = [ScoredReview(review_text=r.review_text, app_id=r.app_id, relevance_score=round(row["relevance_score"], 4),
                                is_relevant=row["is_relevant"], score_band=row["score_band"], rejected_by=row["rejected_by"])
                   for r, row in zip(req.reviews, out.iter_rows(named=True))]
        m = request.app.state.model
        return ScoreResponse(model_run=getattr(m, "run_name", None) or "", threshold=m.threshold, results=results)

    @app.post("/phase1/filter", dependencies=[Depends(_check_key)])
    def filter_games(games: dict[str, GameIn], request: Request,
                     top_k: int | None = Query(None, ge=1, description="keep at most this many reviews per game"),
                     min_score: float | None = Query(None, ge=0, le=1, description="default: the model threshold"),
                     include_scored: bool = Query(True, description="also return every review with its score")
                     ) -> dict[str, dict[str, Any]]:
        if not games:
            raise HTTPException(422, "body must contain at least one game")
        if sum(len(g.reviews) for g in games.values()) > MAX_REVIEWS_PER_REQUEST:
            raise HTTPException(413, f"at most {MAX_REVIEWS_PER_REQUEST} reviews per request")
        store: ContextStore = request.app.state.contexts
        cut = min_score if min_score is not None else request.app.state.model.threshold

        keys, texts, ctxs, names = [], [], [], {}
        for key, g in games.items():
            name, ctx = store.resolve(g, f"game {key!r}")
            names[key] = g.app_name or name
            for t in g.reviews:
                keys.append(key)
                texts.append(t)
                ctxs.append(ctx)
        out = score_rows(texts, ctxs).with_columns(pl.Series("_key", keys))

        result: dict[str, dict[str, Any]] = {}
        for key, g in games.items():
            rows = out.filter(pl.col("_key") == key).sort("relevance_score", descending=True, maintain_order=True)
            kept = rows.filter(pl.col("relevance_score") >= cut)
            if top_k:
                kept = kept.head(top_k)
            extra = {k: v for k, v in (g.model_extra or {}).items()}
            game_out = {**extra, "app_id": g.app_id, "app_name": names[key],
                        "reviews": kept["review_text"].to_list(),
                        "relevance_scores": [round(s, 4) for s in kept["relevance_score"].to_list()]}
            if include_scored:
                game_out["scored"] = [{"review_text": r["review_text"], "relevance_score": round(r["relevance_score"], 4),
                                       "is_relevant": r["is_relevant"], "score_band": r["score_band"],
                                       "rejected_by": r["rejected_by"]}
                                      for r in rows.iter_rows(named=True)]
            result[key] = game_out
        return result

    return app


app = create_app()
