"""Tests for the IBM AML HI adapter."""

from pathlib import Path

import polars as pl
import pytest
from pytest import MonkeyPatch

from amlgraphx.datasets import (
    IBMAML,
    DatasetSource,
    LabelLevel,
    TaskType,
)


@pytest.mark.parametrize(
    ("variant", "transaction", "accounts", "patterns"),
    [
        (
            "hi-small",
            "HI-Small_Trans.csv",
            "HI-Small_accounts.csv",
            "HI-Small_Patterns.txt",
        ),
        (
            "hi-medium",
            "HI-Medium_Trans.csv",
            "HI-Medium_accounts.csv",
            "HI-Medium_Patterns.txt",
        ),
    ],
)
def test_variant_metadata_and_file_mapping(
    variant: str, transaction: str, accounts: str, patterns: str
) -> None:
    """Each IBM variant maps to its explicit source filenames."""
    dataset = IBMAML(variant)

    assert dataset.metadata.name == "IBM AML HI"
    assert dataset.metadata.repo_id == "OsamaMIT/IBM-AML-HI"
    assert dataset.metadata.license == "CDLA-Sharing-1.0"
    assert dataset.metadata.task_type is TaskType.TRANSACTION_CLASSIFICATION
    assert dataset.metadata.label_level is LabelLevel.TRANSACTION
    assert dataset.metadata.source is DatasetSource.HUGGING_FACE
    assert dataset.metadata.expected_files == (transaction, accounts, patterns)


def test_invalid_variant() -> None:
    """Unsupported variants fail at construction time."""
    with pytest.raises(ValueError, match="Unknown IBM AML variant"):
        IBMAML("li-small")


def test_lazy_transaction_and_account_scans(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    """Transactions and accounts are loaded lazily from local synthetic files."""
    transaction = tmp_path / "HI-Small_Trans.csv"
    transaction.write_text(
        "Timestamp,From Bank,Account,Is Laundering\n"
        "2022/09/01 00:20,10,8000EBD30,0\n",
        encoding="utf-8",
    )
    accounts = tmp_path / "HI-Small_accounts.csv"
    accounts.write_text("Account,Is Laundering\n8000EBD30,0\n", encoding="utf-8")
    (tmp_path / "HI-Small_Patterns.txt").write_text("pattern\n", encoding="utf-8")

    monkeypatch.setattr(
        "amlgraphx.datasets.download.snapshot_download", lambda **_: str(tmp_path)
    )
    dataset = IBMAML("hi-small", cache_dir=tmp_path / "cache")
    dataset.download()

    transactions = dataset.transactions()
    accounts_scan = dataset.accounts()
    assert isinstance(transactions, pl.LazyFrame)
    assert isinstance(accounts_scan, pl.LazyFrame)
    assert transactions.collect().height == 1
    assert accounts_scan.collect().height == 1
    assert dataset.transaction_path() == transaction
    assert dataset.accounts_path() == accounts
    assert dataset.patterns_path() == tmp_path / "HI-Small_Patterns.txt"


def test_data_access_requires_download() -> None:
    """File helpers require a completed download."""
    with pytest.raises(RuntimeError, match=r"call download\(\) first"):
        IBMAML("hi-small").transactions()
