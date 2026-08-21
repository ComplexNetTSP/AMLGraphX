"""Tests for the Hugging Face downloader."""

from pathlib import Path
from typing import Any

from pytest import MonkeyPatch

from amlgraphx.datasets import DEFAULT_CACHE_ROOT, HuggingFaceDownloader


def test_default_cache_path() -> None:
    """The default cache is under the user's AMLGraphX cache root."""
    downloader = HuggingFaceDownloader("owner/repo")

    assert downloader.cache_dir == DEFAULT_CACHE_ROOT
    assert downloader.local_dir is None


def test_custom_cache_and_local_paths(tmp_path: Path) -> None:
    """Custom cache and local directories are retained as Paths."""
    cache_dir = tmp_path / "cache"
    local_dir = tmp_path / "dataset"
    downloader = HuggingFaceDownloader(
        "owner/repo", cache_dir=cache_dir, local_dir=local_dir
    )

    assert downloader.cache_dir == cache_dir
    assert downloader.local_dir == local_dir


def test_download_constructs_snapshot_arguments(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    """Downloader forwards repository options without a token argument."""
    captured: dict[str, Any] = {}
    result = tmp_path / "snapshot"

    def fake_snapshot_download(**kwargs: Any) -> str:
        captured.update(kwargs)
        return str(result)

    monkeypatch.setattr(
        "amlgraphx.datasets.download.snapshot_download", fake_snapshot_download
    )
    downloader = HuggingFaceDownloader(
        "owner/repo",
        revision="v1",
        allow_patterns=("*.csv",),
        cache_dir=tmp_path / "cache",
        local_dir=tmp_path / "local",
    )

    assert downloader.download() == result.resolve()
    assert captured == {
        "repo_id": "owner/repo",
        "revision": "v1",
        "allow_patterns": ("*.csv",),
        "cache_dir": tmp_path / "cache",
        "local_dir": tmp_path / "local",
    }
