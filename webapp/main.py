from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from movie_intel.model import MovieRatingModel, predict_rating, recommend_release_slots
from movie_intel.paths import DEFAULT_MERGED, DEFAULT_MODEL, MODELS_DIR

APP_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

app = FastAPI(title="Movie Intel", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")


class PredictIn(BaseModel):
    title: str = Field(..., min_length=1)
    release_date: str
    release_hour: int = 19
    director: str = ""
    cast: str = ""
    genres: str = "Drama"
    budget_usd: float | None = None
    runtime_min: int | None = 120
    language: str = "tr"
    country: str = "TR"


class RecommendIn(PredictIn):
    window_start: str
    window_end: str
    top_n: int = 8


def _load_model() -> MovieRatingModel:
    if not DEFAULT_MODEL.exists():
        raise HTTPException(status_code=503, detail="Model henüz eğitilmedi. `movie-intel train` çalıştırın.")
    return MovieRatingModel.load(DEFAULT_MODEL)


def _metrics() -> dict:
    path = DEFAULT_MODEL.with_suffix(".metrics.json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "metrics": _metrics(),
            "model_ready": DEFAULT_MODEL.exists(),
            "data_ready": DEFAULT_MERGED.exists(),
        },
    )


@app.get("/api/status")
def status() -> dict:
    return {
        "model_ready": DEFAULT_MODEL.exists(),
        "data_ready": DEFAULT_MERGED.exists(),
        "metrics": _metrics(),
        "models_dir": str(MODELS_DIR),
    }


@app.post("/api/predict")
def api_predict(payload: PredictIn) -> dict:
    model = _load_model()
    movie = payload.model_dump()
    rating = predict_rating(model, movie)
    return {"predicted_rating": round(rating, 3), "movie": movie}


@app.post("/api/recommend")
def api_recommend(payload: RecommendIn) -> dict:
    model = _load_model()
    movie = payload.model_dump()
    slots = recommend_release_slots(
        model,
        movie,
        payload.window_start,
        payload.window_end,
        top_n=payload.top_n,
    )
    return {"slots": slots, "best": slots[0] if slots else None}
