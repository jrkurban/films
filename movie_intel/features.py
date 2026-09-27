"""Takvim, bütçe ve hedef kodlama özellikleri — eğitim sızıntısı olmadan."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from movie_intel.schema import TOP_GENRES

CALENDAR_COLUMNS = [
    "year",
    "month",
    "dayofweek",
    "release_hour",
    "is_weekend",
    "is_friday_evening",
    "season",
    "is_awards_window",
    "is_summer",
    "is_january_dump",
]

NUMERIC_BASE = [
    "runtime_min",
    "budget_log",
    "vote_count_log",
]

ENCODED_COLUMNS = [
    "director_enc",
    "actor_1_enc",
    "actor_2_enc",
    "actor_3_enc",
    "language_enc",
    "country_enc",
]


@dataclass
class SmoothedTargetEncoder:
    columns: list[str]
    smoothing: float = 18.0
    maps: dict[str, dict[str, float]] = field(default_factory=dict)
    global_mean: float = 6.5

    def fit(self, frame: pd.DataFrame, target: pd.Series) -> "SmoothedTargetEncoder":
        self.global_mean = float(target.mean())
        self.maps = {}
        for col in self.columns:
            work = pd.DataFrame({"key": frame[col].astype("string").fillna(""), "y": target})
            stats = work.groupby("key", dropna=False)["y"].agg(["mean", "count"])
            encoded = (stats["count"] * stats["mean"] + self.smoothing * self.global_mean) / (
                stats["count"] + self.smoothing
            )
            self.maps[col] = encoded.to_dict()
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=frame.index)
        for col in self.columns:
            mapping = self.maps.get(col, {})
            keys = frame[col].astype("string").fillna("")
            out[f"{col}_enc"] = keys.map(lambda key: mapping.get(str(key), self.global_mean)).astype("float32")
        return out


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    dates = pd.to_datetime(out.get("release_date"), errors="coerce")
    if "year" in out.columns:
        out["year"] = pd.to_numeric(out["year"], errors="coerce")
        out["year"] = out["year"].fillna(dates.dt.year)
    else:
        out["year"] = dates.dt.year
    out["month"] = dates.dt.month
    out["dayofweek"] = dates.dt.dayofweek
    hour = pd.to_numeric(out.get("release_hour"), errors="coerce")
    out["release_hour"] = hour
    out["is_weekend"] = (out["dayofweek"] >= 5).astype("float32")
    out["is_friday_evening"] = ((out["dayofweek"] == 4) & (hour.fillna(-1).between(17, 21))).astype("float32")
    month = out["month"]
    out["season"] = np.select(
        [month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8])],
        [0, 1, 2],
        default=3,
    ).astype("float32")
    out["is_awards_window"] = month.isin([11, 12]).astype("float32")
    out["is_summer"] = month.isin([6, 7]).astype("float32")
    out["is_january_dump"] = month.isin([1, 2]).astype("float32")
    return out


def add_numeric_transforms(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    budget = pd.to_numeric(out.get("budget_usd"), errors="coerce")
    votes = pd.to_numeric(out.get("vote_count"), errors="coerce")
    runtime = pd.to_numeric(out.get("runtime_min"), errors="coerce")
    out["budget_log"] = np.log1p(budget.clip(lower=0)).astype("float32")
    out["vote_count_log"] = np.log1p(votes.clip(lower=0)).astype("float32")
    out["runtime_min"] = runtime.astype("float32")
    return out


def split_cast(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    parts = out.get("cast", pd.Series([""] * len(out))).fillna("").astype(str).str.split(",")
    for index in range(3):
        out[f"actor_{index + 1}"] = parts.map(lambda items, i=index: items[i].strip() if len(items) > i else "")
    out["director"] = out.get("director", "").fillna("").astype(str).str.split(",").str[0].str.strip()
    out["language"] = out.get("language", "").fillna("").astype(str)
    out["country"] = out.get("country", "").fillna("").astype(str)
    return out


def genre_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    genres = frame.get("genres", pd.Series([""] * len(frame))).fillna("").astype(str)
    data = {}
    for genre in TOP_GENRES:
        data[f"genre_{genre.lower().replace('-', '_')}"] = genres.str.contains(genre, regex=False).astype("float32")
    return pd.DataFrame(data, index=frame.index)


def feature_columns() -> list[str]:
    genre_cols = [f"genre_{genre.lower().replace('-', '_')}" for genre in TOP_GENRES]
    return CALENDAR_COLUMNS + NUMERIC_BASE + ENCODED_COLUMNS + genre_cols


class FeatureBuilder:
    def __init__(self) -> None:
        self.encoder = SmoothedTargetEncoder(
            columns=["director", "actor_1", "actor_2", "actor_3", "language", "country"]
        )
        self.columns = feature_columns()

    def fit_transform(self, frame: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        prepared = self._prepare(frame)
        encoded = self.encoder.fit(prepared, target).transform(prepared)
        return self._combine(prepared, encoded)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        prepared = self._prepare(frame)
        encoded = self.encoder.transform(prepared)
        return self._combine(prepared, encoded)

    def _prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        return split_cast(add_numeric_transforms(add_calendar_features(frame)))

    def _combine(self, prepared: pd.DataFrame, encoded: pd.DataFrame) -> pd.DataFrame:
        genres = genre_matrix(prepared)
        combined = pd.concat(
            [prepared.loc[:, [col for col in CALENDAR_COLUMNS + NUMERIC_BASE if col in prepared.columns]], encoded, genres],
            axis=1,
        )
        for col in self.columns:
            if col not in combined.columns:
                combined[col] = np.nan
        return combined.loc[:, self.columns].astype("float32")
