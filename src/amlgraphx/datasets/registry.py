"""Dataset name registry used by the public loader."""

from collections.abc import Mapping

from .base import Dataset
from .ibm_aml import IBMAML
from .paysim import PaySim
from .samld import SAML

DATASET_REGISTRY: Mapping[str, type[Dataset]] = {
    "ibm-aml": IBMAML,
    "paysim": PaySim,
    "saml-d": SAML,
}


def resolve_dataset_class(name: str) -> type[Dataset]:
    """Resolve a public dataset name to its adapter class.

    Args:
        name: Dataset registry name.

    Returns:
        The registered dataset adapter class.

    Raises:
        ValueError: If the name is not registered.
    """
    normalized = name.lower().replace("_", "-")
    aliases = {"ibm": "ibm-aml", "ibm-aml-data": "ibm-aml"}
    normalized = aliases.get(normalized, normalized)
    try:
        return DATASET_REGISTRY[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(f"Unknown dataset {name!r}; choose {supported}") from exc
