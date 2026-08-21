"""Unified dataset loading entry point."""

from pathlib import Path

from .base import Dataset
from .ibm_aml import IBMAML
from .registry import resolve_dataset_class


def load_dataset(
    name: str,
    *,
    variant: str | None = None,
    revision: str = "main",
    cache_dir: Path | None = None,
    local_dir: Path | None = None,
) -> Dataset:
    """Create, cache, and prepare a named AMLGraphX dataset.

    Args:
        name: Registry name such as ``ibm-aml``, ``paysim``, or ``saml-d``.
        variant: Required IBM AML variant.
        revision: Hugging Face repository revision.
        cache_dir: Root directory for the AMLGraphX cache.
        local_dir: Optional direct extraction directory.

    Returns:
        A prepared dataset adapter whose data methods load lazily.

    Raises:
        ValueError: If the dataset name or required variant is invalid.
    """
    dataset_class = resolve_dataset_class(name)
    if dataset_class is IBMAML:
        if variant is None:
            raise ValueError("IBM AML requires a variant")
        dataset = dataset_class(
            variant,
            revision=revision,
            cache_dir=cache_dir,
            local_dir=local_dir,
        )
    else:
        if variant is not None:
            raise ValueError(f"Dataset {name!r} does not accept a variant")
        dataset = dataset_class(
            revision=revision,
            cache_dir=cache_dir,
            local_dir=local_dir,
        )
    dataset.download()
    return dataset
