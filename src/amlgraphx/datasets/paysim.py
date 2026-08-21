"""PaySim dataset adapter."""

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
from .download import (
    DEFAULT_CACHE_ROOT,
    HuggingFaceDownloader,
    find_tabular_file,
)

_REPO_ID = "LordNR/AMLGraphX-Paysim"
_ARCHIVE = "paysim.zip"


class PaySim(Dataset):
    """Load PaySim transactions from a third-party HF mirror.

    The mirror is not the original PaySim publisher. Its README identifies
    the original dataset license as CC BY-SA 4.0 and retains the attribution
    requirements. The loader preserves PaySim transaction, account, and fraud
    columns while applying only conservative lazy cleaning.
    """

    def __init__(
        self,
        *,
        revision: str = "main",
        cache_dir: Path | None = None,
        local_dir: Path | None = None,
    ) -> None:
        """Initialize the PaySim loader.

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
        """Return metadata for the PaySim dataset."""
        return DatasetMetadata(
            name="PaySim",
            task_type=TaskType.TRANSACTION_CLASSIFICATION,
            label_level=LabelLevel.TRANSACTION,
            source=DatasetSource.HUGGING_FACE,
            license="CC-BY-SA-4.0",
            repo_id=_REPO_ID,
            expected_files=(_ARCHIVE,),
        )

    def download(self) -> Path:
        """Reuse or download and extract the PaySim archive."""
        root = self._dataset_root()
        if root.exists():
            try:
                find_tabular_file(root, ("log", "paysim"))
                self._root = root
                return root
            except FileNotFoundError:
                pass
        self._root = self._downloader.download(target_dir=root)
        return self._root

    def transaction_path(self) -> Path:
        """Return the extracted PaySim transaction file path."""
        return find_tabular_file(self.download(), ("log", "paysim"))

    def transactions(self) -> pl.LazyFrame:
        """Return lazily scanned and cleaned PaySim transactions."""
        return clean_lazy_frame(
            pl.scan_csv(self.transaction_path()),
            source_column="nameOrig",
            target_column="nameDest",
        )

    def _dataset_root(self) -> Path:
        if self.local_dir is not None:
            return self.local_dir
        return self.cache_dir / "paysim"
