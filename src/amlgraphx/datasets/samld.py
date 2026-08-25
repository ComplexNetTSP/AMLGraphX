"""SAML-D dataset adapter."""

from pathlib import Path

import polars as pl

from .base import (
    Dataset,
    DatasetMetadata,
    DatasetSource,
    LabelLevel,
    TaskType,
    clean_lazy_frame,
)
from .download import DEFAULT_CACHE_ROOT, HuggingFaceDownloader, find_tabular_file
from .schema import normalize_transactions

_REPO_ID = "LordNR/AMLGraphX-SAML-D"
_ARCHIVE = "SAML-D.zip"


class SAML(Dataset):
    """Load SAML-D transaction or edge data from a third-party HF mirror.

    The mirror README identifies the original dataset license as
    CC BY-NC-SA 4.0 and retains the original attribution requirements. The
    loader discovers the first suitable CSV or Parquet file and resolves source
    and target columns dynamically instead of imposing a fixed schema.
    """

    def __init__(
        self,
        *,
        revision: str = "main",
        cache_dir: Path | None = None,
        local_dir: Path | None = None,
    ) -> None:
        """Initialize the SAML-D loader.

        Args:
            revision: Hugging Face repository revision.
            cache_dir: Root directory for the AMLGraphX cache.
            local_dir: Optional direct extraction directory.
        """
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_ROOT).expanduser()
        self.local_dir = Path(local_dir).expanduser() if local_dir else None
        self._root: Path | None = None
        self._downloader = HuggingFaceDownloader(
            _REPO_ID,
            filename=_ARCHIVE,
            revision=revision,
            cache_dir=self.cache_dir,
        )

    @property
    def metadata(self) -> DatasetMetadata:
        """Return metadata for SAML-D."""
        return DatasetMetadata(
            name="SAML-D",
            task_type=TaskType.TRANSACTION_CLASSIFICATION,
            label_level=LabelLevel.TRANSACTION,
            source=DatasetSource.HUGGING_FACE,
            license="CC-BY-NC-SA-4.0",
            repo_id=_REPO_ID,
            expected_files=(_ARCHIVE,),
        )

    def download(self) -> Path:
        """Reuse or download and extract the SAML-D archive."""
        root = self._dataset_root()
        if root.exists():
            try:
                find_tabular_file(root, ("transaction", "edge", "transfer"))
                self._root = root
                return root
            except FileNotFoundError:
                pass
        self._root = self._downloader.download(target_dir=root)
        return self._root

    def transaction_path(self) -> Path:
        """Return the discovered SAML-D transaction or edge file."""
        return find_tabular_file(self.download(), ("transaction", "edge", "transfer"))

    def transactions(self) -> pl.LazyFrame:
        """Return lazily scanned and dynamically cleaned SAML-D data."""
        path = self.transaction_path()
        frame = (
            pl.scan_parquet(path)
            if path.suffix.lower() == ".parquet"
            else pl.scan_csv(path)
        )
        return normalize_transactions(clean_lazy_frame(frame))

    def _dataset_root(self) -> Path:
        if self.local_dir is not None:
            return self.local_dir
        return self.cache_dir / "saml-d"
