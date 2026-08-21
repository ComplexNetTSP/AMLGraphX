"""IBM AML HI dataset adapter."""

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import polars as pl

from .base import Dataset, DatasetMetadata, DatasetSource, LabelLevel, TaskType
from .download import DEFAULT_CACHE_ROOT, HuggingFaceDownloader

IBMAMLVariant = Literal["hi-small", "hi-medium"]

_REPO_ID = "OsamaMIT/IBM-AML-HI"
_LICENSE = "CDLA-Sharing-1.0"
_VARIANT_FILES: Mapping[str, tuple[str, str, str]] = {
    "hi-small": (
        "HI-Small_Trans.csv",
        "HI-Small_accounts.csv",
        "HI-Small_Patterns.txt",
    ),
    "hi-medium": (
        "HI-Medium_Trans.csv",
        "HI-Medium_accounts.csv",
        "HI-Medium_Patterns.txt",
    ),
}


class IBMAML(Dataset):
    """Access IBM AML HI transaction-level data from a third-party HF mirror.

    The source repository is OsamaMIT/IBM-AML-HI and is licensed under
    CDLA-Sharing-1.0. The supported variants are hi-small and hi-medium. The
    expected files are HI-Small_Trans.csv, HI-Small_accounts.csv,
    HI-Small_Patterns.txt, HI-Medium_Trans.csv, HI-Medium_accounts.csv, and
    HI-Medium_Patterns.txt. Labels are at the transaction level for
    classification.
    """

    def __init__(
        self,
        variant: IBMAMLVariant | str,
        *,
        revision: str = "main",
        cache_dir: Path | None = None,
        local_dir: Path | None = None,
    ) -> None:
        """Initialize an IBM AML HI dataset variant.

        Args:
            variant: Dataset variant, either hi-small or hi-medium.
            revision: Hugging Face repository revision to download.
            cache_dir: Optional Hugging Face cache directory.
            local_dir: Optional directory for the downloaded snapshot.

        Raises:
            ValueError: If variant is unsupported.
        """
        if variant not in _VARIANT_FILES:
            supported = ", ".join(_VARIANT_FILES)
            raise ValueError(f"Unknown IBM AML variant {variant!r}; choose {supported}")

        self.variant = variant
        self.revision = revision
        self._root: Path | None = None
        self._downloader = HuggingFaceDownloader(
            _REPO_ID,
            revision=revision,
            allow_patterns=_VARIANT_FILES[variant],
            cache_dir=cache_dir if cache_dir is not None else DEFAULT_CACHE_ROOT,
            local_dir=local_dir,
        )

    @property
    def metadata(self) -> DatasetMetadata:
        """Return metadata for the selected IBM AML HI variant."""
        return DatasetMetadata(
            name="IBM AML HI",
            task_type=TaskType.TRANSACTION_CLASSIFICATION,
            label_level=LabelLevel.TRANSACTION,
            source=DatasetSource.HUGGING_FACE,
            license=_LICENSE,
            repo_id=_REPO_ID,
            expected_files=_VARIANT_FILES[self.variant],
        )

    def download(self) -> Path:
        """Download the selected variant and return its local root directory."""
        self._root = self._downloader.download()
        return self._root

    def transaction_path(self) -> Path:
        """Return the local path to the transaction CSV."""
        return self._file_path(0)

    def accounts_path(self) -> Path:
        """Return the local path to the account CSV."""
        return self._file_path(1)

    def patterns_path(self) -> Path:
        """Return the local path to the pattern text file."""
        return self._file_path(2)

    def transactions(self) -> pl.LazyFrame:
        """Lazily scan the transaction CSV as a Polars frame."""
        return pl.scan_csv(self.transaction_path())

    def accounts(self) -> pl.LazyFrame:
        """Lazily scan the account CSV as a Polars frame."""
        return pl.scan_csv(self.accounts_path())

    def _file_path(self, index: int) -> Path:
        if self._root is None:
            raise RuntimeError("Dataset is not downloaded; call download() first")
        return self._root / _VARIANT_FILES[self.variant][index]
