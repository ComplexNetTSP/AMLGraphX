"""Dataset interfaces, adapters, and loading entry points for AMLGraphX."""

from amlgraphx.data.schema import normalize_transactions

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
    DatasetDownloadError,
    HuggingFaceDownloader,
    extract_zip,
    find_dataset_file,
    find_tabular_file,
    validate_dataset_files,
)
from .ibm_aml import IBMAML, IBMAMLVariant
from .loader import load_dataset
from .paysim import PaySim
from .samld import SAML

__all__ = [
    "DEFAULT_CACHE_ROOT",
    "IBMAML",
    "SAML",
    "Dataset",
    "DatasetDownloadError",
    "DatasetMetadata",
    "DatasetSource",
    "HuggingFaceDownloader",
    "IBMAMLVariant",
    "LabelLevel",
    "PaySim",
    "TaskType",
    "clean_lazy_frame",
    "extract_zip",
    "find_dataset_file",
    "find_tabular_file",
    "load_dataset",
    "normalize_transactions",
    "validate_dataset_files",
]
