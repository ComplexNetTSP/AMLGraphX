"""Tests for the IBM AML adapter."""

from pathlib import Path
from zipfile import ZipFile

import polars as pl
import pytest
from pytest import MonkeyPatch

from amlgraphx.datasets import IBMAML, DatasetSource, LabelLevel, TaskType, load_dataset


@pytest.mark.parametrize(
    ("variant", "archive", "transaction", "accounts", "patterns"),
    [
        (
            "hi-small",
            "IBM-AML-HI-Small.zip",
            "HI-Small_Trans.csv",
            "HI-Small_accounts.csv",
            "HI-Small_Patterns.txt",
        ),
        (
            "hi-medium",
            "IBM-AML-HI-Medium.zip",
            "HI-Medium_Trans.csv",
            "HI-Medium_accounts.csv",
            "HI-Medium_Patterns.txt",
        ),
        (
            "hi-large",
            "IBM-AML-HI-Large.zip",
            "HI-Large_Trans.csv",
            "HI-Large_accounts.csv",
            "HI-Large_Patterns.txt",
        ),
        (
            "li-small",
            "IBM-AML-LI-Small.zip",
            "LI-Small_Trans.csv",
            "LI-Small_accounts.csv",
            "LI-Small_Patterns.txt",
        ),
        (
            "li-medium",
            "IBM-AML-LI-Medium.zip",
            "LI-Medium_Trans.csv",
            "LI-Medium_accounts.csv",
            "LI-Medium_Patterns.txt",
        ),
        (
            "li-large",
            "IBM-AML-LI-Large.zip",
            "LI-Large_Trans.csv",
            "LI-Large_accounts.csv",
            "LI-Large_Patterns.txt",
        ),
    ],
)
def test_variant_metadata_and_mapping(
    variant: str,
    archive: str,
    transaction: str,
    accounts: str,
    patterns: str,
) -> None:
    """Each IBM variant maps to its explicit archive and extracted files."""
    dataset = IBMAML(variant)

    assert dataset.metadata.name == "IBM AML"
    assert dataset.metadata.repo_id == "LordNR/AMLGraphX-IBM-AML"
    assert dataset.metadata.license == "CDLA-Sharing-1.0"
    assert dataset.metadata.task_type is TaskType.TRANSACTION_CLASSIFICATION
    assert dataset.metadata.label_level is LabelLevel.TRANSACTION
    assert dataset.metadata.source is DatasetSource.HUGGING_FACE
    assert dataset.metadata.expected_files == (archive, transaction, accounts, patterns)


def test_invalid_variant() -> None:
    """Unsupported variants fail before any network call."""
    with pytest.raises(ValueError, match="Unknown IBM AML variant"):
        IBMAML("unknown")


def _zip_file(tmp_path: Path, name: str, files: dict[str, str]) -> Path:
    archive = tmp_path / name
    with ZipFile(archive, "w") as zip_file:
        for filename, content in files.items():
            zip_file.writestr(f"payload/{filename}", content)
    return archive


def test_load_dataset_extracts_and_returns_lazy_frames(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    """The public loader prepares IBM files and returns lazy Polars frames."""
    transaction = (
        "Timestamp,From Bank,Account,To Bank,Account.1,Amount Received,Payment Format,Is Laundering\n"
        "2022/09/01 00:20,10,source-a,20,target-a,10.0,Wire,0\n"
        "2022/09/01 00:21,10,,20,target-b,, ,0\n"
    )
    archive = _zip_file(
        tmp_path,
        "IBM-AML-HI-Small.zip",
        {
            "HI-Small_Trans.csv": transaction,
            "HI-Small_accounts.csv": "Account\nsource-a\n",
            "HI-Small_Patterns.txt": "pattern\n",
        },
    )
    calls: list[dict[str, object]] = []

    def fake_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(archive)

    monkeypatch.setattr("amlgraphx.datasets.download.hf_hub_download", fake_download)
    dataset = load_dataset("ibm-aml", variant="hi-small", cache_dir=tmp_path / "cache")

    transactions = dataset.transactions()
    assert isinstance(transactions, pl.LazyFrame)
    result = transactions.collect()
    assert result.height == 1
    assert "Timestamp__raw" in result.columns
    assert result["Account"][0] == "source-a"
    assert len(calls) == 1
    assert dataset.accounts().collect().height == 1
    assert dataset.patterns().collect().height == 1


def test_existing_cache_skips_download(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """A complete extracted cache is reused by a new adapter instance."""
    archive = _zip_file(
        tmp_path,
        "IBM-AML-HI-Small.zip",
        {
            "HI-Small_Trans.csv": "Account,Account.1\na,b\n",
            "HI-Small_accounts.csv": "Account\na\n",
            "HI-Small_Patterns.txt": "pattern\n",
        },
    )
    calls = 0

    def fake_download(**_: object) -> str:
        nonlocal calls
        calls += 1
        return str(archive)

    monkeypatch.setattr("amlgraphx.datasets.download.hf_hub_download", fake_download)
    cache = tmp_path / "cache"
    load_dataset("ibm-aml", variant="hi-small", cache_dir=cache)
    load_dataset("ibm-aml", variant="hi-small", cache_dir=cache)

    assert calls == 1
