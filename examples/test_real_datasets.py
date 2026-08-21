"""Manual smoke test for real AMLGraphX dataset downloads.

Run this file directly when network access and enough disk space are
available. Each dataset is isolated in a temporary cache and removed after
its inspection.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

from amlgraphx.datasets import load_dataset


def main() -> None:
    """Download each requested dataset one at a time and print its schema."""
    cases = (
        ("ibm-aml", "hi-small"),
        ("ibm-aml", "hi-medium"),
        ("ibm-aml", "li-small"),
        ("paysim", None),
        ("saml-d", None),
    )
    for name, variant in cases:
        with TemporaryDirectory(prefix="amlgraphx-real-") as temporary:
            kwargs = {"cache_dir": Path(temporary)}
            if variant is not None:
                dataset = load_dataset(name, variant=variant, **kwargs)
            else:
                dataset = load_dataset(name, **kwargs)
            transactions = dataset.transactions()
            print(f"{name} {variant or ''}".strip())
            print(transactions.collect_schema())
            print(transactions.head(5).collect())


if __name__ == "__main__":
    main()
