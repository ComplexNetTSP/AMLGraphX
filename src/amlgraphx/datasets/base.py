"""Shared abstractions and conservative cleaning for AMLGraphX datasets."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path

import polars as pl
from pydantic import BaseModel, ConfigDict, Field


class TaskType(StrEnum):
    """Supported machine-learning task types."""

    TRANSACTION_CLASSIFICATION = "transaction_classification"


class LabelLevel(StrEnum):
    """Data entity to which labels apply."""

    TRANSACTION = "transaction"


class DatasetSource(StrEnum):
    """Supported dataset distribution sources."""

    HUGGING_FACE = "huggingface"


class DatasetMetadata(BaseModel):
    """Validated descriptive metadata for a dataset."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    task_type: TaskType
    label_level: LabelLevel
    source: DatasetSource
    license: str = Field(min_length=1)
    repo_id: str = Field(min_length=1)
    expected_files: tuple[str, ...] = Field(min_length=1)


class Dataset(ABC):
    """Base interface for downloadable AMLGraphX datasets."""

    @property
    @abstractmethod
    def metadata(self) -> DatasetMetadata:
        """Return metadata describing the dataset."""

    @abstractmethod
    def download(self) -> Path:
        """Download the dataset and return its local root directory."""

    @abstractmethod
    def transactions(self) -> pl.LazyFrame:
        """Return the cleaned transactions as a lazy Polars frame."""


def clean_lazy_frame(
    frame: pl.LazyFrame,
    *,
    source_column: str | None = None,
    target_column: str | None = None,
    timestamp_columns: Sequence[str] | None = None,
) -> pl.LazyFrame:
    """Apply conservative missing-value handling to a lazy tabular frame.

    Empty rows and rows without a known source or target are removed. Numeric
    nulls use median imputation, while missing string values become
    ``UNKNOWN``. String timestamp columns are parsed lazily and retain a raw
    companion column when parsing cannot represent the original value.

    Args:
        frame: Input lazy Polars frame.
        source_column: Optional sender/source column name.
        target_column: Optional receiver/target column name.
        timestamp_columns: Optional timestamp columns to parse.

    Returns:
        A lazily cleaned Polars frame.
    """
    schema = frame.collect_schema()
    columns = list(schema.names())
    if not columns:
        return frame

    non_empty = pl.any_horizontal(
        *(
            pl.col(column).cast(pl.String).str.strip_chars().fill_null("") != ""
            for column in columns
        )
    )
    frame = frame.filter(non_empty)

    source = _resolve_column(columns, source_column, _SOURCE_ALIASES)
    target = _resolve_column(columns, target_column, _TARGET_ALIASES)
    for column in (source, target):
        if column is not None:
            value = pl.col(column).cast(pl.String)
            frame = frame.filter(value.is_not_null() & (value.str.strip_chars() != ""))

    timestamps = set(timestamp_columns or _timestamp_candidates(columns))
    timestamp_expressions: list[pl.Expr] = []
    for column in timestamps:
        if column not in schema or schema[column] != pl.String:
            continue
        raw_column = f"{column}__raw"
        if raw_column not in columns:
            timestamp_expressions.append(pl.col(column).alias(raw_column))
        timestamp_expressions.append(
            pl.col(column).str.to_datetime(strict=False).alias(column)
        )
    if timestamp_expressions:
        frame = frame.with_columns(timestamp_expressions)

    expressions: list[pl.Expr] = []
    for column, dtype in schema.items():
        if column in timestamps and dtype == pl.String:
            continue
        if dtype.is_numeric():
            expressions.append(
                pl.col(column).fill_null(pl.col(column).median()).alias(column)
            )
        elif dtype == pl.String:
            value = pl.col(column)
            expressions.append(
                pl.when(value.is_null() | (value.str.strip_chars() == ""))
                .then(pl.lit("UNKNOWN"))
                .otherwise(value)
                .alias(column)
            )
    return frame.with_columns(expressions) if expressions else frame


_SOURCE_ALIASES = (
    "source",
    "sender",
    "sender account",
    "from",
    "from account",
    "source id",
    "src",
    "nameorig",
    "origin account",
    "origin",
)
_TARGET_ALIASES = (
    "target",
    "receiver",
    "receiver account",
    "to",
    "to account",
    "account.1",
    "account 1",
    "account duplicated 0",
    "target id",
    "dst",
    "namedest",
    "destination account",
    "destination",
)


def _resolve_column(
    columns: Sequence[str], requested: str | None, aliases: Sequence[str]
) -> str | None:
    if requested in columns:
        return requested
    normalized = {_normalize_column(column): column for column in columns}
    for alias in aliases:
        if _normalize_column(alias) in normalized:
            return normalized[_normalize_column(alias)]
    return None


def _normalize_column(column: str) -> str:
    return " ".join(column.lower().replace("_", " ").replace(".", " ").split())


def _timestamp_candidates(columns: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        column
        for column in columns
        if any(
            token in _normalize_column(column)
            for token in ("timestamp", "datetime", "date")
        )
    )
