"""Başlık normalizasyonu, tip küçültme ve alan temizliği."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

import numpy as np
import pandas as pd

from movie_intel.schema import IMDB_NA, NUMERIC_COLUMNS, STRING_COLUMNS

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_SPACE_RE = re.compile(r"\s+")
_MONEY_RE = re.compile(r"[^\d.\-]")
_ARTICLES = frozenset({"the", "a", "an", "el", "la", "le", "les", "der", "die", "das"})


def replace_imdb_nulls(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.replace({IMDB_NA: pd.NA, "\\N": pd.NA, "nan": pd.NA, "": pd.NA})


def normalize_title(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    text = _PUNCT_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    tokens = [tok for tok in text.split(" ") if tok and tok not in _ARTICLES]
    return " ".join(tokens)


def parse_money(value: object) -> float | pd.NA:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return pd.NA
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return pd.NA if number <= 0 else number
    text = str(value).strip()
    if not text or text in {IMDB_NA, "\\N", "-"}:
        return pd.NA
    cleaned = _MONEY_RE.sub("", text.replace(",", ""))
    if not cleaned:
        return pd.NA
    try:
        number = float(cleaned)
    except ValueError:
        return pd.NA
    return pd.NA if number <= 0 else number


def parse_int(value: object, lo: int | None = None, hi: int | None = None) -> float | pd.NA:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return pd.NA
    try:
        number = int(float(str(value).replace(",", "").strip()))
    except (TypeError, ValueError):
        return pd.NA
    if lo is not None and number < lo:
        return pd.NA
    if hi is not None and number > hi:
        return pd.NA
    return number


def parse_rating(value: object) -> float | pd.NA:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return pd.NA
    number = float(number)
    if number < 0 or number > 10:
        return pd.NA
    return number


def normalize_name_list(value: object, limit: int = 5) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).replace("|", ",").replace(";", ",")
    parts = [part.strip() for part in text.split(",") if part.strip() and part.strip() != IMDB_NA]
    return ", ".join(parts[:limit])


def normalize_genres(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).replace("|", ",").replace(";", ",")
    parts = [part.strip().title() for part in text.split(",") if part.strip() and part.strip() != IMDB_NA]
    seen: list[str] = []
    for part in parts:
        if part not in seen:
            seen.append(part)
    return ",".join(seen)


def downcast_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Sayısal kolonları 32-bit'e indirir; tekrarlayan metinleri category yapar."""
    out = frame.copy()
    for col in out.columns:
        series = out[col]
        if pd.api.types.is_float_dtype(series):
            out[col] = pd.to_numeric(series, errors="coerce").astype("float32")
        elif pd.api.types.is_integer_dtype(series):
            out[col] = pd.to_numeric(series, errors="coerce").astype("Int32")
        elif col in {"title", "cast"}:
            continue
        elif series.dtype == object and series.nunique(dropna=True) < max(64, len(series) // 20):
            out[col] = series.astype("category")
    return out


def optimize_unified(frame: pd.DataFrame) -> pd.DataFrame:
    out = replace_imdb_nulls(frame)
    if "title" in out.columns:
        out["title"] = out["title"].astype("string").str.strip()
        out["title_norm"] = out["title"].map(normalize_title)
    if "year" in out.columns:
        out["year"] = pd.to_numeric(out["year"].map(lambda v: parse_int(v, 1870, 2035)), errors="coerce").astype("Int32")
    if "release_hour" in out.columns:
        out["release_hour"] = pd.to_numeric(out["release_hour"].map(lambda v: parse_int(v, 0, 23)), errors="coerce").astype("Int32")
    if "runtime_min" in out.columns:
        out["runtime_min"] = pd.to_numeric(out["runtime_min"].map(lambda v: parse_int(v, 40, 400)), errors="coerce").astype("Int32")
    if "budget_usd" in out.columns:
        out["budget_usd"] = pd.to_numeric(out["budget_usd"].map(parse_money), errors="coerce").astype("float32")
    if "rating" in out.columns:
        out["rating"] = pd.to_numeric(out["rating"].map(parse_rating), errors="coerce").astype("float32")
    if "vote_count" in out.columns:
        out["vote_count"] = pd.to_numeric(out["vote_count"].map(lambda v: parse_int(v, 1, 50_000_000)), errors="coerce").astype("Int32")
    if "director" in out.columns:
        out["director"] = out["director"].map(lambda v: normalize_name_list(v, 2))
    if "cast" in out.columns:
        out["cast"] = out["cast"].map(lambda v: normalize_name_list(v, 5))
    if "genres" in out.columns:
        out["genres"] = out["genres"].map(normalize_genres)
    if "release_date" in out.columns:
        parsed = pd.to_datetime(out["release_date"], errors="coerce")
        out["release_date"] = parsed.dt.strftime("%Y-%m-%d")
        missing_hour = out.get("release_hour")
        if missing_hour is None:
            out["release_hour"] = pd.array([pd.NA] * len(out), dtype="Int32")
        # Tarihten yıl tamamla
        year_from_date = parsed.dt.year
        if "year" in out.columns:
            out["year"] = out["year"].fillna(year_from_date).astype("Int32")
    for col in STRING_COLUMNS:
        if col in out.columns:
            out[col] = out[col].astype("string")
    return downcast_frame(out)


def match_key(title_norm: object, year: object) -> str:
    year_part = "" if pd.isna(year) else str(int(year))
    return f"{title_norm}|{year_part}"


def coalesce_group(rows: Iterable[pd.Series]) -> pd.Series:
    """Aynı film için kaynakları birleştir: dolu alanları koru, reytingi ağırlıklı ortala."""
    frames = list(rows)
    base = frames[0].copy()
    rating_w: list[tuple[float, float]] = []
    for row in frames:
        for col in base.index:
            if col in {"rating", "vote_count"}:
                continue
            if _is_empty(base[col]) and not _is_empty(row[col]):
                base[col] = row[col]
        rating = row.get("rating")
        votes = row.get("vote_count")
        if not _is_empty(rating):
            weight = float(votes) if not _is_empty(votes) else 1.0
            rating_w.append((float(rating), weight))
    if rating_w:
        total_w = sum(w for _, w in rating_w)
        base["rating"] = sum(r * w for r, w in rating_w) / total_w
        vote_vals = [row.get("vote_count") for row in frames if not _is_empty(row.get("vote_count"))]
        base["vote_count"] = int(max(float(v) for v in vote_vals)) if vote_vals else int(total_w)
    sources = sorted({str(row.get("source")) for row in frames if not _is_empty(row.get("source"))})
    base["source"] = "+".join(sources)
    return base


def _is_empty(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return isinstance(value, str) and not value.strip()


def report_memory(frame: pd.DataFrame) -> str:
    bytes_used = int(frame.memory_usage(deep=True).sum())
    return f"{bytes_used / (1024 * 1024):.2f} MB"


def drop_unused(frame: pd.DataFrame, keep: list[str] | None = None) -> pd.DataFrame:
    keep = keep or list(frame.columns)
    present = [col for col in keep if col in frame.columns]
    return frame.loc[:, present]


def numeric_summary(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for col in NUMERIC_COLUMNS:
        if col not in frame.columns:
            continue
        series = pd.to_numeric(frame[col], errors="coerce")
        summary[col] = {
            "count": float(series.notna().sum()),
            "mean": float(series.mean()) if series.notna().any() else float("nan"),
            "nulls": float(series.isna().sum()),
        }
    return summary
