"""Reyting regresyonu ve vizyon zamanlaması önerisi."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from movie_intel.features import FeatureBuilder

DEFAULT_HOURS = (14, 16, 18, 19, 20, 21)


@dataclass
class TrainResult:
    algorithm: str
    metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    n_train: int
    n_test: int
    feature_importances: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "metrics": self.metrics,
            "baseline_metrics": self.baseline_metrics,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "feature_importances": self.feature_importances,
        }


class MovieRatingModel:
    """HistGradientBoosting ana model; Ridge yorumlanabilir taban çizgisi."""

    def __init__(self) -> None:
        self.features = FeatureBuilder()
        self.model = HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.06,
            max_iter=280,
            min_samples_leaf=20,
            l2_regularization=0.15,
            random_state=42,
        )
        self.baseline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("ridge", Ridge(alpha=2.5)),
            ]
        )
        self.predictor: Any = self.model
        self.trained = False
        self.metrics: dict[str, Any] = {}
        self.global_mean: float = 6.5

    def _xy(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        labeled = frame.dropna(subset=["rating"]).copy()
        if labeled.empty:
            raise ValueError("Eğitim için reytingi olan satır yok.")
        y = labeled["rating"].astype("float32")
        return labeled, y

    def _temporal_split(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        years = pd.to_numeric(frame["year"], errors="coerce")
        cutoff = float(years.quantile(0.8))
        train = frame.loc[years < cutoff].copy()
        test = frame.loc[years >= cutoff].copy()
        if len(train) < 200 or len(test) < 50:
            split = int(len(frame) * 0.8)
            train, test = frame.iloc[:split].copy(), frame.iloc[split:].copy()
        return train, test

    def fit(self, frame: pd.DataFrame) -> TrainResult:
        labeled, _ = self._xy(frame)
        train, test = self._temporal_split(labeled)
        y_train = train["rating"].astype("float32")
        y_test = test["rating"].astype("float32")
        x_train = self.features.fit_transform(train, y_train)
        x_test = self.features.transform(test)

        self.model.fit(x_train, y_train)
        self.baseline.fit(x_train, y_train)
        self.global_mean = float(y_train.mean())
        self.trained = True

        pred = np.clip(self.model.predict(x_test), 1.0, 10.0)
        base_pred = np.clip(self.baseline.predict(x_test), 1.0, 10.0)
        metrics = _regression_metrics(y_test, pred)
        baseline_metrics = _regression_metrics(y_test, base_pred)
        if baseline_metrics["rmse"] < metrics["rmse"]:
            self.predictor = self.baseline
            algorithm = "Ridge"
            chosen = baseline_metrics
            other = metrics
            other_name = "hgb"
        else:
            self.predictor = self.model
            algorithm = "HistGradientBoostingRegressor"
            chosen = metrics
            other = baseline_metrics
            other_name = "ridge"
        importances = _permutation_importance(self.model, x_test, y_test)
        self.metrics = {
            "primary": chosen,
            "comparison": {other_name: other},
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "cutoff_note": "Zamansal ayrım: eski yıllar eğitim, yeni yıllar test.",
        }
        return TrainResult(
            algorithm=algorithm,
            metrics=chosen,
            baseline_metrics=other,
            n_train=len(train),
            n_test=len(test),
            feature_importances=importances,
        )

    def predict_frame(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.trained:
            raise RuntimeError("Model henüz eğitilmedi.")
        features = self.features.transform(frame)
        estimator = getattr(self, "predictor", self.model)
        return np.clip(estimator.predict(features), 1.0, 10.0)

    def predict_one(self, movie: dict[str, Any]) -> float:
        frame = pd.DataFrame([_movie_row(movie)])
        return float(self.predict_frame(frame)[0])

    def recommend_slots(
        self,
        movie: dict[str, Any],
        start: date,
        end: date,
        hours: tuple[int, ...] = DEFAULT_HOURS,
        top_n: int = 8,
    ) -> list[dict[str, Any]]:
        if end < start:
            raise ValueError("Bitiş tarihi başlangıçtan önce olamaz.")
        rows: list[dict[str, Any]] = []
        cursor = start
        while cursor <= end:
            for hour in hours:
                payload = dict(movie)
                payload["release_date"] = cursor.isoformat()
                payload["release_hour"] = hour
                payload["year"] = cursor.year
                pred = self.predict_one(payload)
                rows.append(
                    {
                        "release_date": cursor.isoformat(),
                        "release_hour": hour,
                        "weekday": cursor.strftime("%A"),
                        "predicted_rating": round(pred, 3),
                    }
                )
            cursor += timedelta(days=1)
        rows.sort(key=lambda item: item["predicted_rating"], reverse=True)
        return rows[:top_n]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "MovieRatingModel":
        model = joblib.load(path)
        if not isinstance(model, MovieRatingModel):
            raise TypeError("Geçersiz model dosyası.")
        return model


def _movie_row(movie: dict[str, Any]) -> dict[str, Any]:
    release = movie.get("release_date") or movie.get("release")
    hour = movie.get("release_hour", movie.get("hour", 19))
    year = movie.get("year")
    if release and not year:
        try:
            year = int(str(release)[:4])
        except ValueError:
            year = None
    return {
        "title": movie.get("title", "Untitled"),
        "year": year,
        "release_date": release,
        "release_hour": hour,
        "runtime_min": movie.get("runtime_min", movie.get("runtime")),
        "budget_usd": movie.get("budget_usd", movie.get("budget")),
        "genres": movie.get("genres", ""),
        "director": movie.get("director", ""),
        "cast": movie.get("cast", movie.get("actors", "")),
        "language": movie.get("language", ""),
        "country": movie.get("country", ""),
        "vote_count": movie.get("vote_count"),
        "rating": np.nan,
    }


def _regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": round(rmse, 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def _permutation_importance(
    model: HistGradientBoostingRegressor,
    features: pd.DataFrame,
    target: pd.Series,
    rounds: int = 3,
) -> dict[str, float]:
    rng = np.random.default_rng(0)
    baseline = mean_squared_error(target, model.predict(features))
    scores: dict[str, float] = {}
    values = features.to_numpy(copy=True)
    for index, name in enumerate(features.columns):
        losses = []
        for _ in range(rounds):
            shuffled = values.copy()
            rng.shuffle(shuffled[:, index])
            shuffled_frame = pd.DataFrame(shuffled, columns=features.columns, index=features.index)
            losses.append(mean_squared_error(target, model.predict(shuffled_frame)))
        scores[name] = round(float(np.mean(losses) - baseline), 5)
    ranked = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)[:12])
    return ranked


def predict_rating(model: MovieRatingModel | Path, movie: dict[str, Any]) -> float:
    """Örnek tahmin fonksiyonu: film özellikleri + vizyon anı → tahmini reyting."""
    loaded = model if isinstance(model, MovieRatingModel) else MovieRatingModel.load(Path(model))
    return loaded.predict_one(movie)


def recommend_release_slots(
    model: MovieRatingModel | Path,
    movie: dict[str, Any],
    window_start: str,
    window_end: str,
    hours: tuple[int, ...] = DEFAULT_HOURS,
    top_n: int = 8,
) -> list[dict[str, Any]]:
    """Aday tarih/saat ızgarasında en yüksek tahmini reytingi üreten vizyon dilimlerini döner."""
    loaded = model if isinstance(model, MovieRatingModel) else MovieRatingModel.load(Path(model))
    start = date.fromisoformat(window_start)
    end = date.fromisoformat(window_end)
    return loaded.recommend_slots(movie, start, end, hours=hours, top_n=top_n)


def parse_datetime(value: str) -> datetime:
    text = value.strip()
    if "T" in text:
        return datetime.fromisoformat(text)
    if " " in text:
        return datetime.fromisoformat(text.replace(" ", "T"))
    return datetime.fromisoformat(f"{text}T19:00:00")
