"""Komut satırı: örnek veri, birleştirme, eğitim, tahmin, sunucu."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from movie_intel.merge import load_merged, merge_sources
from movie_intel.model import MovieRatingModel, parse_datetime, predict_rating, recommend_release_slots
from movie_intel.paths import DEFAULT_MERGED, DEFAULT_MODEL, PROCESSED_DIR, RAW_DIR, ensure_dirs
from movie_intel.sample_data import write_sample_sources

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Film verisi birleştirme ve reyting tahmini")


@app.command("generate-sample")
def generate_sample(
    n_movies: int = typer.Option(6000, help="Üretilecek film sayısı"),
    dest: Path = typer.Option(RAW_DIR, help="Ham veri klasörü"),
) -> None:
    paths = write_sample_sources(n_movies=n_movies, dest=dest)
    typer.echo("Örnek kaynaklar yazıldı:")
    for path in paths:
        typer.echo(f"  {path}")


@app.command("merge")
def merge_cmd(
    inputs: Optional[list[Path]] = typer.Argument(None, help="CSV/TSV/Parquet dosyaları"),
    output: Path = typer.Option(DEFAULT_MERGED, help="Birleşik parquet çıktısı"),
    chunksize: int = typer.Option(50_000, help="Parça boyutu"),
    from_sample: bool = typer.Option(False, "--from-sample", help="data/raw altındaki örnekleri kullan"),
) -> None:
    ensure_dirs()
    files = list(inputs or [])
    if from_sample or not files:
        files = sorted(RAW_DIR.rglob("*"))
        files = [path for path in files if path.suffix.lower() in {".csv", ".tsv", ".parquet"} or path.name.endswith(".gz")]
    if not files:
        raise typer.BadParameter("Girdi dosyası yok. Önce generate-sample çalıştırın veya dosya verin.")
    report = merge_sources(files, output, chunksize=chunksize, staging_dir=PROCESSED_DIR / "staging")
    typer.echo(json.dumps(report.as_dict(), indent=2, ensure_ascii=False))


@app.command("train")
def train_cmd(
    data: Path = typer.Option(DEFAULT_MERGED, help="Birleşik parquet"),
    model_path: Path = typer.Option(DEFAULT_MODEL, help="Model kayıt yolu"),
) -> None:
    frame = load_merged(data)
    model = MovieRatingModel()
    result = model.fit(frame)
    model.save(model_path)
    metrics_path = model_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(result.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    typer.echo(f"Model kaydedildi: {model_path}")
    typer.echo(json.dumps(result.as_dict(), indent=2, ensure_ascii=False))


@app.command("predict")
def predict_cmd(
    title: str = typer.Option(..., help="Film adı"),
    release: str = typer.Option(..., help="Vizyon tarihi/saati, örn. 2026-11-20T19:00"),
    director: str = typer.Option("", help="Yönetmen"),
    cast: str = typer.Option("", help="Oyuncular, virgülle"),
    genres: str = typer.Option("Drama", help="Türler"),
    budget: Optional[float] = typer.Option(None, help="Bütçe (USD)"),
    runtime: Optional[int] = typer.Option(120, help="Süre (dk)"),
    language: str = typer.Option("tr", help="Dil kodu"),
    country: str = typer.Option("TR", help="Ülke"),
    model_path: Path = typer.Option(DEFAULT_MODEL, help="Eğitilmiş model"),
) -> None:
    when = parse_datetime(release)
    movie = {
        "title": title,
        "director": director,
        "cast": cast,
        "genres": genres,
        "budget_usd": budget,
        "runtime_min": runtime,
        "language": language,
        "country": country,
        "release_date": when.date().isoformat(),
        "release_hour": when.hour,
        "year": when.year,
    }
    rating = predict_rating(model_path, movie)
    typer.echo(json.dumps({"title": title, "release": when.isoformat(timespec="minutes"), "predicted_rating": round(rating, 3)}, ensure_ascii=False, indent=2))


@app.command("recommend")
def recommend_cmd(
    title: str = typer.Option(..., help="Film adı"),
    window_start: str = typer.Option(..., help="Pencere başlangıcı YYYY-MM-DD"),
    window_end: str = typer.Option(..., help="Pencere bitişi YYYY-MM-DD"),
    director: str = typer.Option("", help="Yönetmen"),
    cast: str = typer.Option("", help="Oyuncular"),
    genres: str = typer.Option("Drama", help="Türler"),
    budget: Optional[float] = typer.Option(None, help="Bütçe (USD)"),
    runtime: Optional[int] = typer.Option(120, help="Süre (dk)"),
    language: str = typer.Option("tr", help="Dil"),
    country: str = typer.Option("TR", help="Ülke"),
    top_n: int = typer.Option(8, help="Kaç slot önerilsin"),
    model_path: Path = typer.Option(DEFAULT_MODEL, help="Eğitilmiş model"),
) -> None:
    movie = {
        "title": title,
        "director": director,
        "cast": cast,
        "genres": genres,
        "budget_usd": budget,
        "runtime_min": runtime,
        "language": language,
        "country": country,
    }
    slots = recommend_release_slots(model_path, movie, window_start, window_end, top_n=top_n)
    typer.echo(json.dumps({"title": title, "slots": slots}, ensure_ascii=False, indent=2))


@app.command("serve")
def serve_cmd(
    host: str = "127.0.0.1",
    port: int = 43147,
    bootstrap: bool = typer.Option(True, help="Model yoksa örnek veri üret, birleştir ve eğit"),
) -> None:
    import uvicorn

    if bootstrap and not DEFAULT_MODEL.exists():
        typer.echo("Model bulunamadı; örnek veri ve eğitim hazırlanıyor...")
        if not any(RAW_DIR.rglob("*.csv")) and not any(RAW_DIR.rglob("*.tsv")):
            write_sample_sources()
        files = [
            path
            for path in RAW_DIR.rglob("*")
            if path.suffix.lower() in {".csv", ".tsv", ".parquet"}
        ]
        if not DEFAULT_MERGED.exists():
            merge_sources(files, DEFAULT_MERGED, staging_dir=PROCESSED_DIR / "staging")
        model = MovieRatingModel()
        result = model.fit(load_merged(DEFAULT_MERGED))
        model.save(DEFAULT_MODEL)
        DEFAULT_MODEL.with_suffix(".metrics.json").write_text(
            json.dumps(result.as_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        typer.echo(json.dumps(result.as_dict(), indent=2, ensure_ascii=False))

    uvicorn.run("webapp.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()
