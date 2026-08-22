"""Tests for cache, extraction, cleaning, and the unified loader."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from zipfile import ZipFile

import polars as pl
import pytest
from pytest import MonkeyPatch

from amlgraphx.datasets import (
    SAML,
    DatasetDownloadError,
    PaySim,
    clean_lazy_frame,
    extract_zip,
    load_dataset,
)
from amlgraphx.graphs import build_account_graph, build_transaction_graph


def test_extract_zip_rejects_invalid_archive(tmp_path: Path) -> None:
    """Invalid ZIP input raises a clear dataset error."""
    archive = tmp_path / "invalid.zip"
    archive.write_text("not a zip", encoding="utf-8")

    with pytest.raises(DatasetDownloadError, match="Invalid ZIP archive"):
        extract_zip(archive, tmp_path / "out")


def test_cleaning_removes_bad_edges_and_imputes_values() -> None:
    """Cleaning removes invalid endpoints and fills conservative nulls."""
    frame = pl.LazyFrame(
        {
            "source": ["a", None, "", "b", None],
            "target": ["x", "y", "z", "q", None],
            "amount": [1.0, 2.0, 3.0, None, None],
            "category": ["known", None, "", None, None],
        }
    )

    result = clean_lazy_frame(
        frame, source_column="source", target_column="target"
    ).collect()

    assert result.height == 2
    assert result["amount"].to_list() == [1.0, 1.0]
    assert result["category"].to_list() == ["known", "UNKNOWN"]


def _make_archive(tmp_path: Path, filename: str, csv_name: str) -> Path:
    archive = tmp_path / filename
    with ZipFile(archive, "w") as zip_file:
        zip_file.writestr(
            f"root/{csv_name}",
            "step,type,amount,nameOrig,nameDest,isFraud\n"
            "1,PAYMENT,1,a,b,0\n",
        )
    return archive


def test_paysim_and_samld_loaders_use_dynamic_tabular_files(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    """PaySim and SAML adapters load their packaged tabular files lazily."""
    paysim_archive = _make_archive(
        tmp_path, "paysim.zip", "PS_20174392719_1491204439457_log.csv"
    )
    samld_archive = _make_archive(tmp_path, "SAML-D.zip", "transactions.csv")
    archives = [paysim_archive, samld_archive]

    def fake_download(**_: object) -> str:
        return str(archives.pop(0))

    monkeypatch.setattr("amlgraphx.datasets.download.hf_hub_download", fake_download)
    cache = tmp_path / "cache"

    paysim = load_dataset("paysim", cache_dir=cache)
    samld = load_dataset("saml-d", cache_dir=cache)

    assert isinstance(paysim, PaySim)
    assert isinstance(samld, SAML)
    paysim_transactions = paysim.transactions()
    assert isinstance(paysim_transactions, pl.LazyFrame)
    assert {"type", "amount", "nameOrig", "nameDest", "isFraud"} <= set(
        paysim_transactions.collect_schema().names()
    )
    assert isinstance(samld.transactions(), pl.LazyFrame)


def test_paysim_transactions_support_both_graph_views(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    """PaySim skips macOS resource files and exposes a logical timestamp."""
    archive = tmp_path / "paysim.zip"
    with ZipFile(archive, "w") as zip_file:
        zip_file.writestr(
            "paysim.csv",
            "step,type,amount,nameOrig,nameDest,isFraud\n"
            "1,PAYMENT,1,A,B,0\n"
            "2,PAYMENT,2,B,C,1\n",
        )
        zip_file.writestr("__MACOSX/._paysim.csv", "not,a,csv\n")

    monkeypatch.setattr(
        "amlgraphx.datasets.download.hf_hub_download",
        lambda **_: str(archive),
    )
    transactions = PaySim(cache_dir=tmp_path / "cache").transactions()

    assert transactions.collect()["timestamp"].to_list() == [
        datetime(1970, 1, 1, 1, tzinfo=UTC),
        datetime(1970, 1, 1, 2, tzinfo=UTC),
    ]
    assert build_account_graph(transactions).num_edges == 2
    assert build_transaction_graph(
        transactions,
        delta=timedelta(hours=1),
    ).num_edges == 1
