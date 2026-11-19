"""Steam Store API fetcher (fallback for app_ids missing from the Kaggle sources).

Rules: one app_id per request, `&l=english`, a configurable delay (default 1.5 s), exponential back-off on
HTTP 429/403, and every response (including success:false) cached in a JSONL file keyed by app_id. A cached
app_id is never requested again, and the cache is appended line by line so a crash loses nothing.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests
from tqdm import tqdm

from src.phase1.utils.io import resolve
from src.phase1.utils.logging import get_logger

log = get_logger("steam_api")

API_URL = "https://store.steampowered.com/api/appdetails"


class ApiBlocked(RuntimeError):
    """Raised when the API keeps answering 429/403 after all retries."""


def load_cache(path: Path) -> dict[int, dict]:
    cache: dict[int, dict] = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    cache[int(row["app_id"])] = row
    return cache


def _append_cache(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def _request(app_id: int, api_cfg: dict, session: requests.Session) -> dict:
    params = {"appids": app_id, "l": api_cfg["language"]}
    for attempt in range(api_cfg["max_retries"]):
        resp = session.get(API_URL, params=params, timeout=30)
        if resp.status_code in (429, 403):
            wait = api_cfg["backoff_base_s"] * (2 ** attempt)
            log.warning("HTTP %d for app %d, backing off %.0fs (attempt %d)", resp.status_code, app_id, wait, attempt + 1)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    raise ApiBlocked(f"app {app_id}: still HTTP 429/403 after {api_cfg['max_retries']} retries")


def parse_response(app_id: int, payload: dict) -> dict:
    """Reduce an appdetails payload to the cached row. `success: false` rows are kept (negative cache)."""
    entry = payload.get(str(app_id), {})
    if not entry.get("success"):
        return {"app_id": app_id, "success": False, "genres": [], "categories": [], "short_description": ""}
    data = entry.get("data", {})
    return {
        "app_id": app_id,
        "success": True,
        "name": data.get("name", ""),
        "genres": [g["description"] for g in data.get("genres", [])],
        "categories": [c["description"] for c in data.get("categories", [])],
        "short_description": data.get("short_description", ""),
    }


def fetch_missing(app_ids: list[int], cfg: dict) -> dict[int, dict]:
    """Fetch app_ids that are not cached yet; return the cache (all app_ids that have an entry)."""
    api_cfg = cfg["context"]["steam_api"]
    cache_path = resolve(api_cfg["cache_file"])
    cache = load_cache(cache_path)
    todo = [a for a in app_ids if a not in cache]
    log.info("steam api: %d requested, %d cached, %d to fetch", len(app_ids), len(app_ids) - len(todo), len(todo))
    if not todo:
        return cache
    session = requests.Session()
    try:
        for app_id in tqdm(todo, desc="steam api"):
            row = parse_response(app_id, _request(app_id, api_cfg, session))
            cache[app_id] = row
            _append_cache(cache_path, row)
            time.sleep(api_cfg["delay_s"])
    except (ApiBlocked, KeyboardInterrupt, requests.RequestException) as e:
        log.error("stopped early (%s); %d fetched so far are cached, rerun to resume", e, len(cache))
    return cache
