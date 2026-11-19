"""Config loading, path helpers, parquet IO and run metadata."""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path

import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[3]


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path = "configs/phase1.yaml") -> dict:
    """Load a YAML config. A config may set `extends: other.yaml` (relative to its own folder)."""
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("extends", None)
    if parent:
        cfg = _deep_merge(load_config(path.parent / parent), cfg)
    return cfg


def resolve(path: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def stage_path(cfg: dict, stage: str, name: str) -> Path:
    """Path of `name` in one of the data stages: interim | processed | annotation | outputs."""
    folder = resolve(cfg["paths"][f"{stage}_dir"])
    folder.mkdir(parents=True, exist_ok=True)
    return folder / name


def run_dir(cfg: dict, run_name: str) -> Path:
    """models/phase1/<run_name>/ (created)."""
    folder = resolve(cfg["paths"]["outputs_dir"]) / run_name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def write_parquet(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def read_parquet(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the previous pipeline step first.")
    return pl.read_parquet(path)


ANNOTATION_SEP = "|"  # rare in reviews (0.2%) and easy to type in Excel's Text to Columns "Other" box
_SEP_STANDIN = "¦"    # replaces the separator inside text, so every line has the same number of columns


def write_annotation_file(df: pl.DataFrame, path: Path, sep: str = ANNOTATION_SEP) -> None:
    """Table for annotators: one line per row, columns split by `sep`, no quotes, UTF-8 with BOM.

    Written as .txt on purpose: Excel splits a .csv on the Windows list separator (`;` on en-ID) as soon as it opens it,
    and descriptions contain `;`. A .txt opens in the import wizard instead (Delimited, Other = `sep`).
    Line breaks and tabs in text become spaces; `sep` inside text becomes `¦`.
    """
    clean = df.with_columns(pl.col(pl.String).str.replace_all(r"[\r\n\t]+", " ")
                            .str.replace_all(sep, _SEP_STANDIN, literal=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    clean.write_csv(path, separator=sep, quote_style="never", include_bom=True, float_precision=4)


def read_annotation_file(path: Path) -> pl.DataFrame:
    """Read an annotator file back, all columns as strings: this .txt format, or the same table re-saved from Excel
    as CSV (`,` or `;`) or as tab-delimited text. The separator is taken from the header line."""
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        header = f.readline()
    sep = max([ANNOTATION_SEP, "\t", ";", ","], key=header.count)
    return pl.read_csv(path, separator=sep, quote_char=None if sep == ANNOTATION_SEP else '"', infer_schema_length=0,
                       encoding="utf8-lossy")


def git_hash() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def save_run_meta(run_dir: Path, cfg: dict, extra: dict | None = None) -> None:
    """Record config + git hash + seed in a run folder."""
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {"seed": cfg["seed"], "git_hash": git_hash(), "config": cfg, **(extra or {})}
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def base_parser(description: str) -> argparse.ArgumentParser:
    """Common CLI: every data/model script takes --config. All other settings live in the yaml."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="configs/phase1.yaml")
    return parser


def config_from_args(args: argparse.Namespace) -> dict:
    return load_config(args.config)
