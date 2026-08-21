"""Hugging Face dataset download helpers."""

from collections.abc import Sequence
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_CACHE_ROOT = Path("~/.cache/amlgraphx").expanduser()


class HuggingFaceDownloader:
    """Download a Hugging Face repository snapshot to a local directory."""

    def __init__(
        self,
        repo_id: str,
        *,
        revision: str = "main",
        allow_patterns: Sequence[str] | None = None,
        cache_dir: Path = DEFAULT_CACHE_ROOT,
        local_dir: Path | None = None,
    ) -> None:
        """Configure a Hugging Face snapshot download.

        Args:
            repo_id: Hugging Face repository identifier.
            revision: Repository branch, tag, or commit.
            allow_patterns: Optional file patterns to include.
            cache_dir: Hugging Face cache directory.
            local_dir: Optional direct destination directory.
        """
        self.repo_id = repo_id
        self.revision = revision
        self.allow_patterns = allow_patterns
        self.cache_dir = Path(cache_dir).expanduser()
        self.local_dir = Path(local_dir).expanduser() if local_dir else None

    def download(self) -> Path:
        """Download the configured snapshot and return its local directory."""
        downloaded = snapshot_download(
            repo_id=self.repo_id,
            revision=self.revision,
            allow_patterns=self.allow_patterns,
            cache_dir=self.cache_dir,
            local_dir=self.local_dir,
        )
        return Path(downloaded).expanduser().resolve()
