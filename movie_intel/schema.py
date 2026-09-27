"""Ortak film şeması — tüm kaynaklar bu kolonlara indirgenir."""

from __future__ import annotations

from typing import Final

UNIFIED_COLUMNS: Final[list[str]] = [
    "movie_id",
    "imdb_id",
    "title",
    "title_norm",
    "year",
    "release_date",
    "release_hour",
    "runtime_min",
    "budget_usd",
    "genres",
    "director",
    "cast",
    "language",
    "country",
    "rating",
    "vote_count",
    "source",
]

NUMERIC_COLUMNS: Final[list[str]] = [
    "year",
    "release_hour",
    "runtime_min",
    "budget_usd",
    "rating",
    "vote_count",
]

STRING_COLUMNS: Final[list[str]] = [
    "movie_id",
    "imdb_id",
    "title",
    "title_norm",
    "genres",
    "director",
    "cast",
    "language",
    "country",
    "source",
    "release_date",
]

# IMDb TSV dosyalarında eksik değer işareti
IMDB_NA: Final[str] = r"\N"

# Kaynak dosya türleri ve beklenen kolon eşlemeleri
COLUMN_ALIASES: Final[dict[str, str]] = {
    "tconst": "imdb_id",
    "primarytitle": "title",
    "originaltitle": "title",
    "startyear": "year",
    "runtimeminutes": "runtime_min",
    "averagerating": "rating",
    "numvotes": "vote_count",
    "directors": "director",
    "movie": "title",
    "film": "title",
    "name": "title",
    "release_year": "year",
    "released": "release_date",
    "release": "release_date",
    "premiere": "release_date",
    "runtime": "runtime_min",
    "budget": "budget_usd",
    "production_budget": "budget_usd",
    "genre": "genres",
    "stars": "cast",
    "actors": "cast",
    "cast_names": "cast",
    "avg_rating": "rating",
    "imdb_rating": "rating",
    "votes": "vote_count",
    "original_language": "language",
    "spoken_language": "language",
    "production_country": "country",
    "country_of_origin": "country",
}

TOP_GENRES: Final[list[str]] = [
    "Action",
    "Adventure",
    "Animation",
    "Biography",
    "Comedy",
    "Crime",
    "Drama",
    "Fantasy",
    "Horror",
    "Romance",
    "Sci-Fi",
    "Thriller",
]
