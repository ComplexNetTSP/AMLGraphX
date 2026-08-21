"""Shared abstractions for AMLGraphX datasets."""

from abc import ABC, abstractmethod
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class TaskType(StrEnum):
    """Supported machine-learning task types."""

    TRANSACTION_CLASSIFICATION = "transaction_classification"


class LabelLevel(StrEnum):
    """Data entity to which labels apply."""

    TRANSACTION = "transaction"


class DatasetSource(StrEnum):
    """Supported dataset distribution sources."""

    HUGGING_FACE = "huggingface"


class DatasetMetadata(BaseModel):
    """Validated descriptive metadata for a dataset."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    task_type: TaskType
    label_level: LabelLevel
    source: DatasetSource
    license: str = Field(min_length=1)
    repo_id: str = Field(min_length=1)
    expected_files: tuple[str, ...] = Field(min_length=1)


class Dataset(ABC):
    """Base interface for downloadable AMLGraphX datasets."""

    @property
    @abstractmethod
    def metadata(self) -> DatasetMetadata:
        """Return metadata describing the dataset."""

    @abstractmethod
    def download(self) -> Path:
        """Download the dataset and return its local root directory."""
