"""Film veri birleştirme, temizlik ve reyting tahmini."""

from movie_intel.model import MovieRatingModel, predict_rating, recommend_release_slots

__all__ = [
    "MovieRatingModel",
    "predict_rating",
    "recommend_release_slots",
]
__version__ = "0.1.0"
