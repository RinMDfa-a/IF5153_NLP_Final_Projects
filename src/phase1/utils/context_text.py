"""The game-context string used for training AND inference (pure Python, no heavy imports).

    Genres: Action; FPS. Categories: Multi-player, Online PvP. Description: <short_description>

Keep this identical between training and inference: the model was trained on exactly this format.
"""
from __future__ import annotations


def truncate_description(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:.") + "..."


def filter_categories(categories: list[str], low_info: list[str]) -> list[str]:
    """Drop low-information categories ("Steam Cloud", "Steam Trading Cards", ...), case-insensitive."""
    drop = {c.lower() for c in low_info}
    return [c for c in categories if c.lower() not in drop]


def make_context_text(genres: list[str], categories: list[str], description: str) -> str:
    parts = [f"Genres: {'; '.join(genres)}."]
    if categories:
        parts.append(f"Categories: {', '.join(categories)}.")
    parts.append(f"Description: {description}")
    return " ".join(parts)


def split_semicolon(value) -> list[str]:
    """'Action; FPS' -> ['Action', 'FPS']. Lists pass through; None / NaN -> []."""
    if value is None or (isinstance(value, float) and value != value):
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [p.strip() for p in str(value).split(";") if p.strip()]
