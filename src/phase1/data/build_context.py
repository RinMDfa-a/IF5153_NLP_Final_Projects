"""Step 3 (BEAM: Transformation, "Join by app_id"): membuat satu string konteks per game.

Urutan prioritas per app_id (sumber yang ngisi tiap field dicatat di log):
  genres / categories: nikdavis steam.csv -> artermiloff -> Steam API
  description: artermiloff short_description -> nikdavis steam_description_data.csv -> Steam API
app_id yang genres atau description-nya masih kosong dibuang dari Task 1 dan dihitung. Categories opsional
(bisa kosong setelah category yang kurang informatif dibuang).

Steam API cuma dipanggil buat app_id yang masih kosong setelah sumber Kaggle dipakai (lihat steam_api.py).
"""
from __future__ import annotations

import polars as pl
from src.phase1.data.steam_api import fetch_missing
from src.phase1.utils.context_text import filter_categories, make_context_text, truncate_description
from src.phase1.utils.io import base_parser, config_from_args, read_parquet, resolve, stage_path, write_parquet
from src.phase1.utils.logging import get_logger, log_rows
from src.phase1.utils.seed import set_seed
from src.phase1.utils.text_clean import parse_list, parse_semicolon, strip_html

log = get_logger("build_context")
CONTEXT_COLUMNS = ["app_id", "name", "genres", "categories", "description", "context_text", "src_genres", "src_description"]

# --- readers: satu per sumber ----------------------------------------------------------------------------
def _read_csv(path_str: str, columns: list[str]) -> pl.DataFrame:
    path = resolve(path_str)
    if not path.exists():
        log.warning("%s not found; this source is skipped", path)
        return pl.DataFrame({c: [] for c in columns}, schema={c: pl.String for c in columns})
    df = pl.read_csv(path, infer_schema_length=0, truncate_ragged_lines=True, quote_char='"')
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"{path.name} lacks columns {missing}; header is {df.columns}")
    return df.select(columns)


def load_nikdavis(cfg: dict) -> dict[int, dict]:
    src = cfg["context"]["sources"]["nikdavis"]
    df = _read_csv(src["steam_csv"], ["appid", "name", "genres", "categories"])
    out = {}
    for row in df.iter_rows(named=True):
        out[int(row["appid"])] = {
            "name": row["name"] or "",
            "genres": parse_semicolon(row["genres"]),
            "categories": parse_semicolon(row["categories"]),
        }
    return out


def load_nikdavis_descriptions(cfg: dict) -> dict[int, str]:
    src = cfg["context"]["sources"]["nikdavis"]
    df = _read_csv(src["description_csv"], ["steam_appid", "short_description"])
    return {int(r["steam_appid"]): strip_html(r["short_description"]) for r in df.iter_rows(named=True)}


def load_artermiloff(cfg: dict) -> dict[int, dict]:
    src = cfg["context"]["sources"]["artermiloff"]
    fmt = src["list_format"]
    df = _read_csv(src["csv"], ["appid", "name", "short_description", "genres", "categories"])
    out, failed = {}, 0
    for row in df.iter_rows(named=True):
        try:
            genres, categories = parse_list(row["genres"], fmt), parse_list(row["categories"], fmt)
        except (ValueError, SyntaxError):
            failed += 1
            genres, categories = [], []
        out[int(row["appid"])] = {
            "name": row["name"] or "",
            "genres": genres,
            "categories": categories,
            "short_description": strip_html(row["short_description"]),
        }
    if failed:
        log.warning("artermiloff: %d rows did not parse as list_format=%r; check context.sources.artermiloff.list_format",
                    failed, fmt)
    return out


def inspect_sources(cfg: dict, n: int = 3) -> None:
    """Print header tiap sumber dan beberapa cell list mentah. Jalankan sekali sebelum percaya sama parser-nya."""
    srcs = cfg["context"]["sources"]
    for label, path_str in [("nikdavis steam.csv", srcs["nikdavis"]["steam_csv"]),
                            ("nikdavis steam_description_data.csv", srcs["nikdavis"]["description_csv"]),
                            ("artermiloff", srcs["artermiloff"]["csv"]),
                            ("reviews", cfg["reviews"]["file"])]:
        path = resolve(path_str)
        if not path.exists():
            print(f"[{label}] {path} NOT FOUND")
            continue
        head = pl.read_csv(path, n_rows=n, infer_schema_length=0, truncate_ragged_lines=True)
        print(f"[{label}] columns: {head.columns}")
        for col in ("genres", "categories", "tags"):
            if col in head.columns:
                print(f"   {col}: {head[col].to_list()}")


# --- rakit ---------------------------------------------------------------------------------------------
def build_game_context(app_ids: list[int], names: dict[int, str], cfg: dict) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (game_context, coverage_report) untuk app_id review yang diberikan."""
    ccfg = cfg["context"]
    nik, nik_desc, art = load_nikdavis(cfg), load_nikdavis_descriptions(cfg), load_artermiloff(cfg)

    info: dict[int, dict] = {a: {"genres": [], "categories": [], "description": "", "src_genres": None,
                                 "src_description": None, "name": names.get(a, "")} for a in app_ids}
    for a in app_ids:
        rec = info[a]
        for src_name, table in (("nikdavis", nik), ("artermiloff", art)):
            if a in table and table[a]["genres"] and not rec["genres"]:
                rec["genres"], rec["categories"], rec["src_genres"] = table[a]["genres"], table[a]["categories"], src_name
            if a in table and table[a]["name"]:
                rec["name"] = rec["name"] or table[a]["name"]
        if a in art and art[a]["short_description"]:
            rec["description"], rec["src_description"] = art[a]["short_description"], "artermiloff"
        elif nik_desc.get(a):
            rec["description"], rec["src_description"] = nik_desc[a], "nikdavis"

    still_missing = [a for a in app_ids if not info[a]["genres"] or not info[a]["description"]]
    api_tried = 0
    if still_missing and ccfg["steam_api"]["enabled"]:
        api_tried = len(still_missing)
        cache = fetch_missing(still_missing, cfg)
        for a in still_missing:
            row = cache.get(a)
            if not row or not row.get("success"):
                continue
            rec = info[a]
            if not rec["genres"] and row["genres"]:
                rec["genres"], rec["categories"], rec["src_genres"] = row["genres"], row["categories"], "api"
            if not rec["description"] and row["short_description"]:
                rec["description"], rec["src_description"] = strip_html(row["short_description"]), "api"
            rec["name"] = rec["name"] or row.get("name", "")

    rows = []
    for a in app_ids:
        rec = info[a]
        if not rec["genres"] or not rec["description"]:
            continue
        cats = filter_categories(rec["categories"], ccfg["low_info_categories"])
        desc = truncate_description(rec["description"], ccfg["max_description_chars"])
        rows.append({"app_id": a, "name": rec["name"], "genres": rec["genres"], "categories": cats, "description": desc,
                     "context_text": make_context_text(rec["genres"], cats, desc),
                     "src_genres": rec["src_genres"], "src_description": rec["src_description"]})
    context = pl.DataFrame(rows, schema={"app_id": pl.Int64, "name": pl.String, "genres": pl.List(pl.String),
                                         "categories": pl.List(pl.String), "description": pl.String,
                                         "context_text": pl.String, "src_genres": pl.String, "src_description": pl.String})
    report = coverage_report(app_ids, info, context, api_tried)
    log_rows(log, "build_context (app_ids)", len(app_ids), context.height)
    return context.select(CONTEXT_COLUMNS), report


def coverage_report(app_ids: list[int], info: dict[int, dict], context: pl.DataFrame, api_tried: int) -> pl.DataFrame:
    """Porsi app_id review yang diisi oleh tiap sumber, oleh API, dan yang dibuang."""
    n = max(len(app_ids), 1)
    kept = set(context["app_id"].to_list())
    rows = []
    for field in ("src_genres", "src_description"):
        counts: dict[str, int] = {}
        for a in app_ids:
            counts[info[a][field] or "missing"] = counts.get(info[a][field] or "missing", 0) + 1
        for source, c in sorted(counts.items()):
            rows.append({"field": field, "source": source, "n_app_ids": c, "share": c / n})
    dropped = len([a for a in app_ids if a not in kept])
    rows.append({"field": "overall", "source": "dropped (missing genres or description)", "n_app_ids": dropped, "share": dropped / n})
    rows.append({"field": "overall", "source": "kept", "n_app_ids": len(kept), "share": len(kept) / n})
    rows.append({"field": "overall", "source": f"sent to steam api ({api_tried})", "n_app_ids": api_tried, "share": api_tried / n})
    return pl.DataFrame(rows)


def main() -> None:
    cfg = config_from_args(base_parser("Step 3: bikin konteks game").parse_args())
    set_seed(cfg["seed"])
    reviews = read_parquet(stage_path(cfg, "interim", "reviews_filtered.parquet"))
    names = dict(reviews.group_by("app_id").agg(pl.col("app_name").first()).iter_rows())
    app_ids = sorted(names)
    context, report = build_game_context(app_ids, names, cfg)
    write_parquet(context, stage_path(cfg, "interim", "game_context.parquet"))
    with pl.Config(tbl_rows=50, fmt_str_lengths=60):
        print(report)


if __name__ == "__main__":
    main()
