"""Tests for the canonical transaction schema."""

import polars as pl
import pytest

from amlgraphx.datasets import normalize_transactions


def test_normalize_transactions_adds_canonical_columns() -> None:
    """Raw dataset names become shared columns without losing raw data."""
    frame = pl.LazyFrame(
        {
            "Account": [" A "],
            "Account.1": ["B"],
            "Timestamp": ["2025-01-01"],
            "Amount Received": [10.0],
            "Is Laundering": [1],
        }
    )

    result = normalize_transactions(
        frame,
        source_column="Account",
        target_column="Account.1",
        timestamp_column="Timestamp",
        amount_column="Amount Received",
        label_column="Is Laundering",
    ).collect()

    assert result.select(
        ["transaction_id", "source", "target", "timestamp", "amount", "label"]
    ).to_dicts() == [
        {
            "transaction_id": "tx_0",
            "source": "A",
            "target": "B",
            "timestamp": "2025-01-01",
            "amount": 10.0,
            "label": 1,
        }
    ]
    assert "Account" in result.columns


def test_normalize_transactions_requires_endpoints() -> None:
    """A canonical transaction must identify both accounts."""
    with pytest.raises(ValueError, match="source and target"):
        normalize_transactions(pl.LazyFrame({"amount": [1.0]}))


def test_normalize_transactions_combines_date_and_time() -> None:
    """A time-only column is combined with its transaction date."""
    result = normalize_transactions(
        pl.LazyFrame(
            {
                "Sender_account": ["A"],
                "Receiver_account": ["B"],
                "Date": ["2025-01-01"],
                "Time": ["10:35:19"],
            }
        )
    ).collect()

    assert result["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").to_list() == [
        "2025-01-01 10:35:19"
    ]
