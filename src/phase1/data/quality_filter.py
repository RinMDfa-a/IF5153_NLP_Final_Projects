"""Step 2 (BEAM: Review Quality Rules -> Quality Filtering): quality filter berbasis rule yang gampang dijelaskan.

Tiap rule adalah fungsi terpisah yang menerima satu teks review dan return True kalau rule-nya kena (fires). Rule jalan
sesuai urutan di `quality.rules`; review ditolak oleh rule pertama yang kena, dan nama rule itu dicatat.
Rule yang murah jalan duluan supaya language detector yang lambat cuma memproses review yang lolos.
"""
from __future__ import annotations

import re
import polars as pl
from src.phase1.utils.io import base_parser, config_from_args, read_parquet, stage_path, write_parquet
from src.phase1.utils.logging import get_logger, log_rows
from src.phase1.utils.seed import set_seed

log = get_logger("quality_filter")

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


# --- satu fungsi per rule -------------------------------------------------------------------------------


def rule_empty_or_placeholder(text: str, cfg: dict) -> bool:
    norm = text.strip().lower()
    return norm == "" or norm in {p.lower() for p in cfg["placeholders"]}


def rule_too_short(text: str, cfg: dict) -> bool:
    return len(_words(text)) < cfg["min_tokens"]


_RUN_RE = re.compile(r"(.)\1+", re.DOTALL)
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)


def nonalnum_ratio(text: str) -> float:
    """Porsi karakter non-alfanumerik dari semua karakter non-spasi, setelah deretan karakter yang sama diringkas jadi satu.
    Diringkas supaya "AWESOME!!!!!!!!!!" atau pemisah "-----" nggak kebaca sebagai ASCII art, sementara
    gambar dari block character (banyak simbol berbeda) tetap skornya mendekati 1."""
    chars = [c for c in _RUN_RE.sub(r"\1", text) if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if not c.isalnum()) / len(chars)


def rule_ascii_art(text: str, cfg: dict) -> bool:
    return len(text) >= cfg["min_chars_ascii_art"] and nonalnum_ratio(text) > cfg["max_nonalnum_ratio"]


def max_char_run(text: str) -> int:
    """Deretan terpanjang dari huruf atau angka yang sama ("sooooooo"). Deretan tanda baca ("!!!!", "-----", sensor
    "♥♥♥♥") nggak dihitung: itu umum di review asli, dan ASCII art sudah ditangani `ascii_art`."""
    best = run = 0
    prev = None
    for c in text:
        run = run + 1 if (c == prev and c.isalnum()) else 1
        prev = c
        best = max(best, run)
    return best


def repeated_char_share(text: str, min_run: int) -> float:
    """Porsi huruf/angka yang ada di dalam deretan karakter sama sepanjang minimal `min_run`."""
    n_alnum = sum(c.isalnum() for c in text)
    if not n_alnum:
        return 0.0
    in_runs = sum(len(m.group(0)) for m in re.finditer(r"([^\W_])\1{%d,}" % (min_run - 1), text))
    return in_runs / n_alnum


def rule_repeated_chars(text: str, cfg: dict) -> bool:
    """Fires kalau karakter yang dipanjang-panjangin ("sooooooooo", "aaaaaaaa") jadi porsi besar dari teks. Satu kata
    yang dipanjangin ("ahhhhhhhh") di dalam review yang selebihnya normal nggak bikin rule ini kena."""
    return max_char_run(text) >= cfg["max_char_run"] and \
        repeated_char_share(text, cfg["max_char_run"]) >= cfg["max_repeated_char_share"]


def unique_word_ratio(text: str) -> float:
    # split di whitespace supaya "10/10 10/10 10/10" kehitung sebagai token yang berulang
    tokens = text.lower().split()
    return len(set(tokens)) / len(tokens) if tokens else 1.0


def rule_repeated_words(text: str, cfg: dict) -> bool:
    tokens = text.split()
    return len(tokens) >= cfg["repeat_min_words"] and unique_word_ratio(text) < cfg["min_unique_word_ratio"]


def rule_non_english(text: str, cfg: dict) -> bool:
    lang_cfg = cfg["language"]
    if not lang_cfg["enabled"]:
        return False
    text = _URL_RE.sub(" ", text).strip()  # link nggak punya sinyal bahasa dan bikin detector condong ke hu / de
    if len(text) < lang_cfg["min_chars"]:  # language ID nggak andal untuk teks yang sangat pendek atau slang
        return False
    head = text[: lang_cfg["max_chars"]]
    if lang_cfg.get("backend", "langdetect") == "fasttext":
        # lowercase: fastText lid salah baca English ALL-CAPS ("UNBELIEVABLY BORING AVOID AT ALL COSTS") sebagai bahasa lain
        lang, prob = detect_language_fasttext(head.lower(), lang_cfg["model_path"])
        # prediksi non-English yang yakin akan fires; tebakan yang nggak yakin pada slang / teks pendek tetap dipertahankan
        return lang != lang_cfg["target"] and prob >= lang_cfg["min_confidence"]
    return detect_language(head) != lang_cfg["target"]


LID_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
_ft_models: dict = {}


def _load_fasttext(model_path: str):
    """Load model language-ID fastText lid.176 (di-download saat pertama kali dipakai)."""
    if model_path not in _ft_models:
        import urllib.request

        import fasttext

        from src.phase1.utils.io import resolve

        path = resolve(model_path)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            log.info("downloading %s to %s", LID_URL, path)
            urllib.request.urlretrieve(LID_URL, path)
        fasttext.FastText.eprint = lambda *a, **k: None  # bungkam warning saat load
        _ft_models[model_path] = fasttext.load_model(str(path))
    return _ft_models[model_path]


def detect_language_fasttext(text: str, model_path: str) -> tuple[str, float]:
    """(kode bahasa, probabilitas). Panggil predict level bawah: wrapper numpy fastText error di numpy 2."""
    model = _load_fasttext(model_path)
    preds = model.f.predict(text.replace("\n", " ") + "\n", 1, 0.0, "strict")
    if not preds:
        return "unknown", 0.0
    prob, label = preds[0]
    return label.replace("__label__", ""), float(prob)


_detector_ready = False


def detect_language(text: str) -> str:
    """langdetect dengan seed tetap (lambat, ~10 ms per review). Return 'unknown' kalau nggak bisa ditentukan."""
    global _detector_ready
    try:
        from langdetect import DetectorFactory, detect
        from langdetect.lang_detect_exception import LangDetectException
    except ImportError as e:
        raise ImportError("langdetect is required for the non_english rule: pip install langdetect "
                          "(or set quality.language.enabled: false)") from e
    if not _detector_ready:
        DetectorFactory.seed = 0
        _detector_ready = True
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


# Rule per baris memetakan teks ke bool; duplicate_in_game butuh seluruh frame jadi ditangani terpisah.
TEXT_RULES = {
    "empty_or_placeholder": rule_empty_or_placeholder,
    "too_short": rule_too_short,
    "ascii_art": rule_ascii_art,
    "repeated_chars": rule_repeated_chars,
    "repeated_words": rule_repeated_words,
    "non_english": rule_non_english,
}


def rule_duplicate_in_game(df: pl.DataFrame) -> pl.Series:
    """True untuk setiap salinan setelah yang pertama dari teks (yang sudah dinormalisasi) yang sama di dalam satu game."""
    norm = pl.col("review_text").str.to_lowercase().str.replace_all(r"\s+", " ").str.strip_chars()
    return df.select((pl.int_range(pl.len()).over([pl.col("app_id"), norm]) > 0).alias("dup"))["dup"]


def review_stats(df: pl.DataFrame, min_run: int = 8) -> pl.DataFrame:
    """Statistik per review buat nentuin threshold (lihat persentilnya di notebook)."""
    texts = df["review_text"].to_list()
    return df.select("review_id").with_columns(
        pl.Series("n_words", [len(_words(t)) for t in texts], dtype=pl.Int32),
        pl.Series("nonalnum_ratio", [nonalnum_ratio(t) for t in texts], dtype=pl.Float32),
        pl.Series("max_char_run", [max_char_run(t) for t in texts], dtype=pl.Int32),
        pl.Series("repeated_char_share", [repeated_char_share(t, min_run) for t in texts], dtype=pl.Float32),
        pl.Series("unique_word_ratio", [unique_word_ratio(t) for t in texts], dtype=pl.Float32),
    )


def apply_rules(df: pl.DataFrame, qcfg: dict) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (kept, rejected). `rejected` punya kolom tambahan `rule_name`."""
    remaining = df
    rejected_parts: list[pl.DataFrame] = []
    for name in qcfg["rules"]:
        if remaining.height == 0:
            break
        if name == "duplicate_in_game":
            mask = rule_duplicate_in_game(remaining)
        elif name in TEXT_RULES:
            fn = TEXT_RULES[name]
            mask = pl.Series([fn(t, qcfg) for t in remaining["review_text"].to_list()], dtype=pl.Boolean)
        else:
            raise KeyError(f"unknown quality rule {name!r}; known: {sorted([*TEXT_RULES, 'duplicate_in_game'])}")
        fired = remaining.filter(mask)
        log.info("rule %-22s fired on %d of %d", name, fired.height, remaining.height)
        if fired.height:
            rejected_parts.append(fired.with_columns(pl.lit(name).alias("rule_name")))
        remaining = remaining.filter(~mask)
    rejected = pl.concat(rejected_parts) if rejected_parts else df.head(0).with_columns(pl.lit("").alias("rule_name"))
    return remaining, rejected


def rule_summary(rejected: pl.DataFrame, n_total: int) -> pl.DataFrame:
    return (rejected.group_by("rule_name").len().rename({"len": "n_rejected"})
            .with_columns((pl.col("n_rejected") / n_total).alias("share_of_input"))
            .sort("n_rejected", descending=True))


def main() -> None:
    cfg = config_from_args(base_parser("Step 2: filter kualitas review").parse_args())
    set_seed(cfg["seed"])
    df = read_parquet(stage_path(cfg, "interim", "reviews.parquet"))
    kept, rejected = apply_rules(df, cfg["quality"])
    log_rows(log, "quality_filter", df.height, kept.height)
    write_parquet(kept, stage_path(cfg, "interim", "reviews_filtered.parquet"))
    write_parquet(rejected, stage_path(cfg, "interim", "reviews_rejected.parquet"))
    print(rule_summary(rejected, df.height))


if __name__ == "__main__":
    main()
