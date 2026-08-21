"""IBM AML dataset adapter."""

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

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
    find_dataset_file,
    validate_dataset_files,
)

IBMAMLVariant = Literal[
    "hi-small",
    "hi-medium",
    "hi-large",
    "li-small",
    "li-medium",
    "li-large",
]

_REPO_ID = "LordNR/AMLGraphX-IBM-AML"
_LICENSE = "CDLA-Sharing-1.0"
_VARIANT_FILES: Mapping[str, tuple[str, str, str, str]] = {
    "hi-small": (
        "IBM-AML-HI-Small.zip",
        "HI-Small_Trans.csv",
        "HI-Small_accounts.csv",
        "HI-Small_Patterns.txt",
    ),
    "hi-medium": (
        "IBM-AML-HI-Medium.zip",
        "HI-Medium_Trans.csv",
        "HI-Medium_accounts.csv",
        "HI-Medium_Patterns.txt",
    ),
    "hi-large": (
        "IBM-AML-HI-Large.zip",
        "HI-Large_Trans.csv",
        "HI-Large_accounts.csv",
        "HI-Large_Patterns.txt",
    ),
    "li-small": (
        "IBM-AML-LI-Small.zip",
        "LI-Small_Trans.csv",
        "LI-Small_accounts.csv",
        "LI-Small_Patterns.txt",
    ),
    "li-medium": (
        "IBM-AML-LI-Medium.zip",
        "LI-Medium_Trans.csv",
        "LI-Medium_accounts.csv",
        "LI-Medium_Patterns.txt",
    ),
    "li-large": (
        "IBM-AML-LI-Large.zip",
        "LI-Large_Trans.csv",
        "LI-Large_accounts.csv",
        "LI-Large_Patterns.txt",
    ),
}


class IBMAML(Dataset):
    """Load IBM AML HI and LI variants from a third-party HF mirror.

    The source repository is ``LordNR/AMLGraphX-IBM-AML`` and uses the
    CDLA-Sharing-1.0 license. Supported variants are hi-small, hi-medium,
    hi-large, li-small, li-medium, and li-large. Each archive is expected to
    contain its ``*_Trans.csv``, ``*_accounts.csv``, and ``*_Patterns.txt``
    files. Transactions are cleaned lazily for transaction classification.
    """

    def __init__(
        self,
        variant: IBMAMLVariant | str,
        *,
        revision: str = "main",
        cache_dir: Path | None = None,
        local_dir: Path | None = None,
    ) -> None:
        """Initialize an IBM AML variant.

        Args:
            variant: One of the supported HI or LI variants.
            revision: Hugging Face repository revision.
            cache_dir: Root directory for the AMLGraphX cache.
            local_dir: Optional direct extraction directory.

        Raises:
            ValueError: If the variant is unsupported.
        """
        if variant not in _VARIANT_FILES:
            supported = ", ".join(_VARIANT_FILES)
            raise ValueError(f"Unknown IBM AML variant {variant!r}; choose {supported}")
        self.variant = variant
        self.revision = revision
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_ROOT).expanduser()
        self.local_dir = Path(local_dir).expanduser() if local_dir else None
        archive, *_ = _VARIANT_FILES[variant]
        self._downloader = HuggingFaceDownloader(
            _REPO_ID,
            filename=archive,
            revision=revision,
            cache_dir=self.cache_dir,
        )
        self._root: Path | None = None

    @property
    def metadata(self) -> DatasetMetadata:
        """Return metadata for the selected IBM AML variant."""
        files = _VARIANT_FILES[self.variant]
        return DatasetMetadata(
            name="IBM AML",
            task_type=TaskType.TRANSACTION_CLASSIFICATION,
            label_level=LabelLevel.TRANSACTION,
            source=DatasetSource.HUGGING_FACE,
            license=_LICENSE,
            repo_id=_REPO_ID,
            expected_files=files,
        )

    def download(self) -> Path:
        """Reuse or download, extract, and validate the selected variant."""
        root = self._dataset_root()
        expected = _VARIANT_FILES[self.variant][1:]
        if root.exists():
            try:
                validate_dataset_files(root, expected)
                self._root = root
                return root
            except FileNotFoundError:
                pass
        self._root = self._downloader.download(
            target_dir=root,
            expected_files=expected,
        )
        return self._root

    def transaction_path(self) -> Path:
        """Return the extracted transaction CSV path."""
        return find_dataset_file(self.download(), _VARIANT_FILES[self.variant][1])

    def accounts_path(self) -> Path:
        """Return the extracted accounts CSV path."""
        return find_dataset_file(self.download(), _VARIANT_FILES[self.variant][2])

    def patterns_path(self) -> Path:
        """Return the extracted patterns text path."""
        return find_dataset_file(self.download(), _VARIANT_FILES[self.variant][3])

    def transactions(self) -> pl.LazyFrame:
        """Return lazily scanned and conservatively cleaned transactions."""
        frame = pl.scan_csv(self.transaction_path())
        return clean_lazy_frame(
            frame,
            source_column="Account",
            target_column="Account.1",
            timestamp_columns=("Timestamp",),
        )

    def accounts(self) -> pl.LazyFrame:
        """Return lazily scanned and cleaned account data."""
        return clean_lazy_frame(pl.scan_csv(self.accounts_path()))

    def patterns(self) -> pl.LazyFrame:
        """Return the patterns text as a one-column lazy frame."""
        return pl.scan_csv(
            self.patterns_path(),
            has_header=False,
            new_columns=["pattern"],
            separator="\t",
            infer_schema=False,
            truncate_ragged_lines=True,
        )

    def _dataset_root(self) -> Path:
        if self.local_dir is not None:
            return self.local_dir
        return self.cache_dir / "ibm-aml" / self.variant
