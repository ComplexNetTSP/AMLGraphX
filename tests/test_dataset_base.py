"""Tests for dataset base abstractions."""

import pytest
from pydantic import ValidationError

from amlgraphx.datasets import DatasetMetadata, DatasetSource, LabelLevel, TaskType


def test_enum_values() -> None:
    """Enums expose stable string values."""
    assert TaskType.TRANSACTION_CLASSIFICATION.value == "transaction_classification"
    assert LabelLevel.TRANSACTION.value == "transaction"
    assert DatasetSource.HUGGING_FACE.value == "huggingface"


def test_metadata_validation() -> None:
    """Valid metadata is accepted and enum strings are coerced."""
    metadata = DatasetMetadata(
        name="example",
        task_type="transaction_classification",
        label_level="transaction",
        source="huggingface",
        license="CDLA-Sharing-1.0",
        repo_id="owner/repo",
        expected_files=("transactions.csv",),
    )

    assert metadata.task_type is TaskType.TRANSACTION_CLASSIFICATION
    assert metadata.expected_files == ("transactions.csv",)


def test_metadata_rejects_missing_or_empty_fields() -> None:
    """Invalid metadata fails Pydantic validation."""
    with pytest.raises(ValidationError):
        DatasetMetadata(
            name="",
            task_type=TaskType.TRANSACTION_CLASSIFICATION,
            label_level=LabelLevel.TRANSACTION,
            source=DatasetSource.HUGGING_FACE,
            license="CDLA-Sharing-1.0",
            repo_id="owner/repo",
            expected_files=(),
        )
