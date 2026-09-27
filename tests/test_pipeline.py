from pathlib import Path

from movie_intel.merge import load_merged, merge_sources
from movie_intel.model import MovieRatingModel, predict_rating, recommend_release_slots
from movie_intel.sample_data import write_sample_sources


def test_merge_train_predict(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    files = write_sample_sources(n_movies=400, dest=raw)
    merged_path = tmp_path / "movies.parquet"
    report = merge_sources(files, merged_path, chunksize=200, staging_dir=tmp_path / "staging")
    assert report.rows_after_merge > 350
    frame = load_merged(merged_path)
    assert frame["rating"].notna().sum() > 200
    assert frame["budget_usd"].notna().sum() > 100

    model = MovieRatingModel()
    result = model.fit(frame)
    assert result.metrics["mae"] < 1.2
    assert result.n_train > 50

    movie = {
        "title": "Deneme",
        "director": "Nuri Bilge Ceylan",
        "cast": "Haluk Bilginer, Tilda Swinton",
        "genres": "Drama",
        "budget_usd": 4_000_000,
        "runtime_min": 128,
        "language": "tr",
        "country": "TR",
        "release_date": "2026-11-20",
        "release_hour": 19,
        "year": 2026,
    }
    rating = predict_rating(model, movie)
    assert 1.0 <= rating <= 10.0

    slots = recommend_release_slots(model, movie, "2026-11-01", "2026-11-10", top_n=5)
    assert len(slots) == 5
    scores = [item["predicted_rating"] for item in slots]
    assert scores == sorted(scores, reverse=True)
