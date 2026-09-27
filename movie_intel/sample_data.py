"""IMDb benzeri ve yayın reytingi kaynakları için sentetik ama gerçekçi örnek veri."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from movie_intel.paths import RAW_DIR, ensure_dirs

DIRECTORS = [
    ("Nuri Bilge Ceylan", 0.92),
    ("Zeki Demirkubuz", 0.78),
    ("Yeşim Ustaoğlu", 0.70),
    ("Christopher Nolan", 0.95),
    ("Greta Gerwig", 0.82),
    ("Bong Joon-ho", 0.94),
    ("Denis Villeneuve", 0.90),
    ("Chloe Zhao", 0.74),
    ("Martin Scorsese", 0.96),
    ("Celine Song", 0.80),
    ("Ryan Coogler", 0.77),
    ("Justine Triet", 0.83),
    ("Park Chan-wook", 0.88),
    ("Sofia Coppola", 0.76),
    ("Asghar Farhadi", 0.86),
]

ACTORS = [
    ("Haluk Bilginer", 0.84),
    ("Tilda Swinton", 0.81),
    ("Cillian Murphy", 0.86),
    ("Song Kang-ho", 0.90),
    ("Emma Stone", 0.83),
    ("Timothée Chalamet", 0.79),
    ("Beren Saat", 0.72),
    ("Florence Pugh", 0.80),
    ("Mahershala Ali", 0.88),
    ("Saoirse Ronan", 0.85),
    ("Kıvanç Tatlıtuğ", 0.68),
    ("Zendaya", 0.75),
    ("Adam Driver", 0.82),
    ("Youn Yuh-jung", 0.87),
    ("Ralph Fiennes", 0.89),
]

GENRE_SETS = [
    ("Drama", 0.18),
    ("Drama,Romance", 0.10),
    ("Action,Thriller", 0.08),
    ("Action,Adventure,Sci-Fi", 0.10),
    ("Comedy", 0.06),
    ("Comedy,Romance", 0.07),
    ("Horror,Thriller", 0.06),
    ("Crime,Drama", 0.08),
    ("Biography,Drama", 0.07),
    ("Animation,Adventure,Family", 0.06),
    ("Fantasy,Adventure", 0.05),
    ("Sci-Fi,Drama", 0.05),
    ("Documentary", -0.02),
    ("Mystery,Thriller", 0.06),
]

TITLES = [
    "Kış Uykusu",
    "Ahlat Ağacı",
    "Kuru Otlar Üstüne",
    "Bir Zamanlar Anadolu'da",
    "Distant Harbor",
    "Night Train to Ankara",
    "The Last Olive Grove",
    "Silent District",
    "Paper Lanterns",
    "Red Clay Summer",
    "Orbit of Dust",
    "The Cartographer",
    "Blue Hour in Izmir",
    "Glass Minaret",
    "Second Language",
    "Harbor Lights",
    "The Auditorium",
    "North Road Motel",
    "A Modest Eclipse",
    "Velvet Checkpoint",
]


def _title_for(index: int, rng: np.random.Generator) -> str:
    if index < len(TITLES):
        return TITLES[index]
    stem = rng.choice(
        [
            "Echo",
            "Corridor",
            "Threshold",
            "Anchor",
            "Mirror",
            "Tide",
            "Ember",
            "Atlas",
            "Signal",
            "Orchard",
        ]
    )
    return f"{stem} {index:04d}"


def _timing_bonus(month: int, hour: int, weekday: int, genres: str) -> float:
    bonus = 0.0
    if weekday >= 5:
        bonus += 0.08
    if weekday == 4 and 17 <= hour <= 21:
        bonus += 0.14
    if month in {1, 2}:
        bonus -= 0.28
    if month in {6, 7} and any(tag in genres for tag in ("Action", "Adventure", "Sci-Fi")):
        bonus += 0.22
    if month in {11, 12} and any(tag in genres for tag in ("Drama", "Biography")):
        bonus += 0.24
    if hour < 10:
        bonus -= 0.16
    if weekday == 1 and hour < 14:
        bonus -= 0.12
    return bonus


def generate_catalog(n_movies: int = 6000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for index in range(n_movies):
        director, director_skill = DIRECTORS[int(rng.integers(0, len(DIRECTORS)))]
        actor_idx = rng.choice(len(ACTORS), size=3, replace=False)
        cast = [ACTORS[int(i)] for i in actor_idx]
        genre_pair = GENRE_SETS[int(rng.integers(0, len(GENRE_SETS)))]
        genres, genre_bonus = genre_pair
        year = int(rng.integers(2008, 2025))
        month = int(rng.integers(1, 13))
        day = int(rng.integers(1, 28))
        hour = int(rng.choice([10, 14, 16, 18, 19, 20, 21], p=[0.08, 0.12, 0.12, 0.18, 0.2, 0.2, 0.1]))
        weekday = int(pd.Timestamp(year=year, month=month, day=day).dayofweek)
        runtime = int(np.clip(rng.normal(118, 18), 80, 190))
        budget = float(np.exp(rng.normal(16.4, 1.15)))
        budget = float(np.clip(budget, 250_000, 220_000_000))
        log_budget = np.log(budget)
        cast_skill = float(np.mean([skill for _, skill in cast]))
        timing = _timing_bonus(month, hour, weekday, genres)
        runtime_bonus = -abs(runtime - 118) / 80
        rating = (
            5.55
            + 1.45 * director_skill
            + 0.55 * cast_skill
            + 0.28 * np.tanh((log_budget - 16.2) / 1.8)
            + genre_bonus
            + 0.35 * runtime_bonus
            + timing
            + float(rng.normal(0, 0.38))
        )
        rating = float(np.clip(rating, 1.4, 9.6))
        votes = int(np.clip(np.exp(rng.normal(8.2 + (rating - 6.5) * 0.45, 1.1)), 80, 1_800_000))
        language = rng.choice(["en", "tr", "ko", "fr", "es"], p=[0.46, 0.22, 0.12, 0.12, 0.08])
        country = {"en": "US", "tr": "TR", "ko": "KR", "fr": "FR", "es": "ES"}[str(language)]
        rows.append(
            {
                "imdb_id": f"tt{8_000_000 + index:07d}",
                "title": _title_for(index, rng),
                "year": year,
                "release_date": f"{year:04d}-{month:02d}-{day:02d}",
                "release_hour": hour,
                "runtime_min": runtime,
                "budget_usd": round(budget, 0),
                "genres": genres,
                "director": director,
                "cast": ", ".join(name for name, _ in cast),
                "language": language,
                "country": country,
                "rating": round(rating, 2),
                "vote_count": votes,
            }
        )
    return pd.DataFrame(rows)


def write_sample_sources(n_movies: int = 6000, dest: Path | None = None) -> list[Path]:
    """IMDb TSV + bütçe CSV + yayınlanmış reyting CSV üretir."""
    ensure_dirs()
    dest = dest or RAW_DIR
    catalog = generate_catalog(n_movies)
    imdb_dir = dest / "imdb"
    box_dir = dest / "boxoffice"
    pub_dir = dest / "published"
    for folder in (imdb_dir, box_dir, pub_dir):
        folder.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(7)
    basics = pd.DataFrame(
        {
            "tconst": catalog["imdb_id"],
            "titleType": "movie",
            "primaryTitle": catalog["title"],
            "originalTitle": catalog["title"],
            "startYear": catalog["year"].astype("string"),
            "runtimeMinutes": catalog["runtime_min"].astype("string"),
            "genres": catalog["genres"].str.replace(",", ",", regex=False),
        }
    )
    drop_runtime = rng.random(len(basics)) < 0.04
    basics.loc[drop_runtime, "runtimeMinutes"] = r"\N"

    ratings = catalog[["imdb_id", "rating", "vote_count"]].rename(
        columns={"imdb_id": "tconst", "rating": "averageRating", "vote_count": "numVotes"}
    )
    # Kullanıcının elindeki yayın reytingleri: alt küme + küçük gürültü
    published_mask = rng.random(len(catalog)) < 0.72
    published = catalog.loc[published_mask, ["title", "year", "rating", "vote_count", "release_date"]].copy()
    published["rating"] = (published["rating"] + rng.normal(0, 0.12, size=len(published))).clip(1, 10)
    published = published.rename(
        columns={
            "title": "film",
            "year": "release_year",
            "rating": "avg_rating",
            "vote_count": "votes",
            "release_date": "premiere",
        }
    )

    crew = pd.DataFrame({"tconst": catalog["imdb_id"], "directors": catalog["director"], "writers": r"\N"})

    # Bütçe kaynağı farklı kolon adlarıyla ve kısmi örtüşmeyle
    budget_mask = rng.random(len(catalog)) < 0.78
    budgets = catalog.loc[budget_mask, ["title", "year", "budget_usd", "director", "cast", "language", "country", "release_hour"]].copy()
    budgets = budgets.rename(
        columns={
            "title": "movie",
            "year": "release_year",
            "budget_usd": "production_budget",
            "cast": "stars",
        }
    )
    # IMDb kimliği olmayan ekstra filmler
    extra = generate_catalog(max(120, n_movies // 20), seed=99)
    extra = extra.head(max(80, n_movies // 25))
    extra_budget = extra.rename(
        columns={
            "title": "movie",
            "year": "release_year",
            "budget_usd": "production_budget",
            "cast": "stars",
        }
    )[["movie", "release_year", "production_budget", "director", "stars", "language", "country", "release_hour"]]
    budgets = pd.concat([budgets, extra_budget], ignore_index=True)

    basics_path = imdb_dir / "title.basics.tsv"
    ratings_path = imdb_dir / "title.ratings.tsv"
    crew_path = imdb_dir / "title.crew.tsv"
    budget_path = box_dir / "budgets.csv"
    published_path = pub_dir / "ratings.csv"

    basics.to_csv(basics_path, sep="\t", index=False)
    ratings.to_csv(ratings_path, sep="\t", index=False)
    crew.to_csv(crew_path, sep="\t", index=False)
    budgets.to_csv(budget_path, index=False)
    published.to_csv(published_path, index=False)
    return [basics_path, ratings_path, crew_path, budget_path, published_path]
