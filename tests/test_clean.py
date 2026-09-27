from movie_intel.clean import normalize_title, parse_money, parse_rating


def test_normalize_title_strips_articles_and_punct() -> None:
    assert normalize_title("The Last Olive Grove!") == "last olive grove"
    assert normalize_title("Kış Uykusu") == "kış uykusu"


def test_parse_money_and_rating() -> None:
    assert parse_money("$1,200,000") == 1_200_000
    assert parse_rating(8.4) == 8.4
    import pandas as pd

    assert pd.isna(parse_rating(11))
