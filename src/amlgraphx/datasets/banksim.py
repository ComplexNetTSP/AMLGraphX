"""BankSim customer-to-merchant transaction dataset adapter."""

from datetime import timedelta
from pathlib import Path

import polars as pl

from amlgraphx.data import logical_timestamp_from_step
from amlgraphx.data.schema import normalize_transactions

from .base import (
    Dataset,
    DatasetMetadata,
    DatasetSource,
    LabelLevel,
    TaskType,
    clean_lazy_frame,
)
from .download import DEFAULT_CACHE_ROOT, HuggingFaceDownloader, find_dataset_file


class BankSim(Dataset):
    """Load BankSim as customer-to-merchant transaction rows."""

    _repo_id = "LordNR/AMLGraphX-Banksim"
    _archive = "Banksim.zip"
    _file = "bs140513_032310.csv"

    def __init__(
        self,
        *,
        revision: str = "main",
        cache_dir: Path | None = None,
        local_dir: Path | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_ROOT).expanduser()
        self.local_dir = Path(local_dir).expanduser() if local_dir else None
        self._downloader = HuggingFaceDownloader(
            self._repo_id,
            filename=self._archive,
            revision=revision,
            cache_dir=self.cache_dir,
        )

    @property
    def metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            name="BankSim",
            task_type=TaskType.TRANSACTION_CLASSIFICATION,
            label_level=LabelLevel.TRANSACTION,
            source=DatasetSource.HUGGING_FACE,
            license="CC-BY-SA-4.0",
            repo_id=self._repo_id,
            expected_files=(self._archive,),
        )

    def download(self) -> Path:
        root = self.local_dir or self.cache_dir / "banksim"
        if not (root / self._file).exists():
            self._downloader.download(target_dir=root)
        return root

    def transactions(self) -> pl.LazyFrame:
        """Return customer→merchant rows with logical daily step timestamps."""
        frame = clean_lazy_frame(
            pl.scan_csv(
                find_dataset_file(self.download(), self._file),
                quote_char="'",
            ),
            source_column="customer",
            target_column="merchant",
        )
        frame = logical_timestamp_from_step(
            frame, step_column="step", step_size=timedelta(days=1)
        )
        return normalize_transactions(
            frame,
            source_column="customer",
            target_column="merchant",
            timestamp_column="timestamp",
            amount_column="amount",
            label_column="fraud",
        )
