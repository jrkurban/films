"""Parça parça okuma ve Parquet yazma — tüm tabloyu RAM'e almak zorunda kalmaz."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from movie_intel.schema import IMDB_NA


def iter_table_chunks(
    path: Path,
    *,
    chunksize: int = 50_000,
    sep: str | None = None,
    columns: list[str] | None = None,
) -> Iterator[pd.DataFrame]:
    """CSV/TSV/Parquet dosyasını parçalar halinde okur. gzip desteklenir."""
    path = Path(path)
    suffix = "".join(path.suffixes).lower()
    if suffix.endswith(".parquet"):
        dataset = ds.dataset(path, format="parquet")
        scanner = dataset.scanner(columns=columns, batch_size=chunksize)
        for batch in scanner.to_batches():
            yield batch.to_pandas()
        return

    inferred_sep = sep or ("\t" if ".tsv" in suffix else ",")
    compression = "gzip" if suffix.endswith(".gz") else "infer"
    reader = pd.read_csv(
        path,
        sep=inferred_sep,
        chunksize=chunksize,
        compression=compression,
        na_values=[IMDB_NA, "\\N", ""],
        keep_default_na=True,
        dtype_backend="numpy_nullable",
        encoding="utf-8",
        encoding_errors="replace",
        low_memory=True,
        usecols=columns,
    )
    for chunk in reader:
        yield chunk


def write_parquet_chunks(chunks: Iterator[pd.DataFrame], dest: Path) -> int:
    """Parçaları tek bir Parquet dosyasına satır grupları olarak yazar."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    rows = 0
    try:
        for chunk in chunks:
            if chunk.empty:
                continue
            table = pa.Table.from_pandas(_stabilize_dtypes(chunk), preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(dest, table.schema, compression="zstd")
            else:
                table = _align_table(table, writer.schema)
            writer.write_table(table)
            rows += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    return rows


def _stabilize_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Tüm-NA kolonların Arrow'da null tipine düşmesini engeller."""
    out = frame.copy()
    for col in out.columns:
        if out[col].isna().all():
            if pd.api.types.is_numeric_dtype(out[col]) or col in {
                "year",
                "release_hour",
                "runtime_min",
                "budget_usd",
                "rating",
                "vote_count",
            }:
                out[col] = pd.Series(np.full(len(out), np.nan), dtype="float32")
            else:
                out[col] = pd.Series([pd.NA] * len(out), dtype="string")
        elif out[col].dtype == object:
            out[col] = out[col].astype("string")
    return out


def _align_table(table: pa.Table, schema: pa.Schema) -> pa.Table:
    arrays = []
    for field in schema:
        if field.name in table.column_names:
            column = table[field.name]
            if pa.types.is_null(field.type):
                arrays.append(column)
            elif pa.types.is_null(column.type):
                arrays.append(pa.nulls(table.num_rows, type=field.type))
            else:
                arrays.append(column.cast(field.type, safe=False))
        else:
            fill_type = pa.string() if pa.types.is_null(field.type) else field.type
            arrays.append(pa.nulls(table.num_rows, type=fill_type))
    names = [field.name for field in schema]
    return pa.Table.from_arrays(arrays, names=names)


def read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    return pq.read_table(path, columns=columns).to_pandas()


def parquet_row_count(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows
