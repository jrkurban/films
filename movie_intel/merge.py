"""Çok kaynaklı film tablolarını ortak şemaya çevirip bellek dostu biçimde birleştirir."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from movie_intel.clean import (
    coalesce_group,
    downcast_frame,
    match_key,
    optimize_unified,
    report_memory,
)
from movie_intel.schema import COLUMN_ALIASES, UNIFIED_COLUMNS
from movie_intel.store import iter_table_chunks, write_parquet_chunks


@dataclass
class MergeReport:
    input_files: list[str]
    rows_ingested: int
    rows_after_merge: int
    memory_before: str
    memory_after: str
    output_path: str

    def as_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


def _rename_columns(frame: pd.DataFrame) -> pd.DataFrame:
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for raw in frame.columns:
        key = raw.strip().lower().replace(" ", "_")
        target = COLUMN_ALIASES.get(key, key)
        if target in UNIFIED_COLUMNS and target not in used:
            mapping[raw] = target
            used.add(target)
    return frame.rename(columns=mapping)


def _ensure_columns(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    out = frame.copy()
    for col in UNIFIED_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    if "source" not in frame.columns or out["source"].isna().all():
        out["source"] = source
    return out.loc[:, UNIFIED_COLUMNS]


def normalize_chunk(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    renamed = _rename_columns(frame)
    unified = _ensure_columns(renamed, source)
    return optimize_unified(unified)


def ingest_file(path: Path, dest: Path, source: str, chunksize: int = 50_000) -> int:
    def chunks() -> Iterator[pd.DataFrame]:
        for raw in iter_table_chunks(path, chunksize=chunksize):
            yield normalize_chunk(raw, source)

    return write_parquet_chunks(chunks(), dest)


def merge_sources(
    inputs: list[Path],
    output: Path,
    *,
    chunksize: int = 50_000,
    staging_dir: Path | None = None,
) -> MergeReport:
    """
    Birden fazla CSV/TSV/Parquet kaynağını ortak şemaya çevirir,
    (normalize başlık + yıl) veya imdb_id üzerinden tekilleştirir.
    """
    if not inputs:
        raise ValueError("Birleştirilecek en az bir dosya gerekli.")

    staging_dir = staging_dir or output.parent / "staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    staging_parts: list[Path] = []
    ingested = 0
    memory_before = "n/a"

    for index, path in enumerate(inputs):
        part = staging_dir / f"part_{index:03d}_{path.stem}.parquet"
        rows = ingest_file(path, part, source=path.stem, chunksize=chunksize)
        ingested += rows
        staging_parts.append(part)

    # Tekilleştirme için anahtarları üretip sıralı tarama yap.
    keyed_path = staging_dir / "keyed.parquet"

    def keyed_chunks() -> Iterator[pd.DataFrame]:
        for part in staging_parts:
            frame = pd.read_parquet(part)
            frame["imdb_id"] = frame["imdb_id"].astype("string")
            frame["dedup_key"] = [
                str(imdb) if pd.notna(imdb) and str(imdb).strip() else match_key(title, year)
                for imdb, title, year in zip(frame["imdb_id"], frame["title_norm"], frame["year"])
            ]
            yield frame

    write_parquet_chunks(keyed_chunks(), keyed_path)
    keyed = pd.read_parquet(keyed_path)
    memory_before = report_memory(keyed)
    merged = _resolve_duplicates(keyed)
    merged = merged.reset_index(drop=True)
    merged["movie_id"] = [f"mv{index:07d}" for index in range(1, len(merged) + 1)]
    merged = downcast_frame(merged.loc[:, UNIFIED_COLUMNS])

    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output, index=False, compression="zstd")

    return MergeReport(
        input_files=[str(path) for path in inputs],
        rows_ingested=ingested,
        rows_after_merge=len(merged),
        memory_before=memory_before,
        memory_after=report_memory(merged),
        output_path=str(output),
    )


def _has_imdb(series: pd.Series) -> pd.Series:
    text = series.astype("string").fillna("").str.strip()
    return text.ne("") & text.ne("<NA>")


def _collapse(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    work = frame.sort_values([key, "vote_count"], ascending=[True, False], na_position="last")
    rows: list[pd.Series] = []
    for _, group in work.groupby(key, sort=False, dropna=False):
        if len(group) == 1:
            rows.append(group.iloc[0])
        else:
            rows.append(coalesce_group(row for _, row in group.iterrows()))
    return pd.DataFrame(rows)


def _resolve_duplicates(frame: pd.DataFrame) -> pd.DataFrame:
    """Önce imdb_id, sonra (başlık + yıl) ile tekilleştir — kaynak şemaları farklı olsa da birleşir."""
    work = frame.copy()
    work["title_key"] = [
        match_key(title, year) for title, year in zip(work["title_norm"], work["year"])
    ]
    flagged = _has_imdb(work["imdb_id"])
    by_id = _collapse(work.loc[flagged], "imdb_id") if flagged.any() else work.iloc[0:0].copy()
    leftover = work.loc[~flagged]
    combined = pd.concat([by_id, leftover], ignore_index=True)
    if "dedup_key" in combined.columns:
        combined = combined.drop(columns=["dedup_key"])
    merged = _collapse(combined, "title_key")
    drop_cols = [col for col in ("dedup_key", "title_key") if col in merged.columns]
    return merged.drop(columns=drop_cols) if drop_cols else merged


def load_merged(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
