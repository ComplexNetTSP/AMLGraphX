"""Elliptic transaction-node graph dataset adapters."""

from datetime import timedelta
from pathlib import Path

import polars as pl

from amlgraphx.graph import TransactionGraph, build_precomputed_transaction_graph

from .base import (
    DatasetMetadata,
    DatasetSource,
    LabelLevel,
    TaskType,
    TransactionGraphDataset,
)
from .download import DEFAULT_CACHE_ROOT, HuggingFaceDownloader, find_dataset_file


class _EllipticBase(TransactionGraphDataset):
    """Shared loader for an Elliptic transaction table and supplied edge list."""

    repo_id: str
    archive: str
    feature_file: str
    class_file: str
    edge_file: str
    features_have_header: bool
    name: str

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
            self.repo_id,
            filename=self.archive,
            revision=revision,
            cache_dir=self.cache_dir,
        )

    @property
    def metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            name=self.name,
            task_type=TaskType.TRANSACTION_CLASSIFICATION,
            label_level=LabelLevel.TRANSACTION,
            source=DatasetSource.HUGGING_FACE,
            license="CC-BY-SA-4.0",
            repo_id=self.repo_id,
            expected_files=(self.archive,),
        )

    def download(self) -> Path:
        root = self._root()
        if not root.exists() or not all(
            (root / file).exists()
            for file in (self.feature_file, self.class_file, self.edge_file)
        ):
            self._downloader.download(target_dir=root)
        return root

    def transaction_nodes(self) -> pl.LazyFrame:
        """Return transaction features joined to the canonical binary label."""
        features = self._features()
        classes = pl.scan_csv(
            find_dataset_file(self.download(), self.class_file)
        ).with_columns(
            pl.col("txId").cast(pl.String),
            pl.when(pl.col("class").cast(pl.String) == "1")
            .then(1)
            .when(pl.col("class").cast(pl.String) == "2")
            .then(0)
            .otherwise(None)
            .cast(pl.Int8)
            .alias("label"),
        )
        return features.with_columns(pl.col("txId").cast(pl.String)).join(
            classes.select("txId", "class", "label"), on="txId", how="left"
        )

    def transaction_edges(self) -> pl.LazyFrame:
        """Return the supplied directed transaction-to-transaction edge list."""
        return pl.scan_csv(find_dataset_file(self.download(), self.edge_file))

    def transaction_graph(
        self, *, step_size: timedelta = timedelta(days=1)
    ) -> TransactionGraph:
        """Return the published transaction graph using logical step time.

        A step is mapped to a configured logical duration; it is not claimed to
        be a wall-clock timestamp.  The original ``Time step`` column remains.
        """
        return build_precomputed_transaction_graph(
            self.transaction_nodes(),
            self.transaction_edges(),
            node_id_column="txId",
            edge_source_column="txId1",
            edge_target_column="txId2",
            time_column="Time step",
            step_size=step_size,
        )

    def _features(self) -> pl.LazyFrame:
        path = find_dataset_file(self.download(), self.feature_file)
        if self.features_have_header:
            return pl.scan_csv(path)
        return pl.scan_csv(path, has_header=False).rename(
            {"column_1": "txId", "column_2": "Time step"}
        )

    def _root(self) -> Path:
        return (
            self.local_dir
            if self.local_dir is not None
            else self.cache_dir / self.name.lower().replace("+", "plus")
        )


class Elliptic(_EllipticBase):
    """Load the original Elliptic precomputed transaction graph."""

    repo_id = "LordNR/AMLGraphX-Elliptic"
    archive = "Elliptic.zip"
    feature_file = "elliptic_bitcoin_dataset/elliptic_txs_features.csv"
    class_file = "elliptic_bitcoin_dataset/elliptic_txs_classes.csv"
    edge_file = "elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv"
    features_have_header = False
    name = "Elliptic"


class EllipticPlusPlus(_EllipticBase):
    """Load Elliptic++'s transaction-node view, excluding address relations."""

    repo_id = "LordNR/AMLGraphX-ElliptiPlusPlus"
    archive = "Elliptic ++.zip"
    feature_file = "txs_features.csv"
    class_file = "txs_classes.csv"
    edge_file = "txs_edgelist.csv"
    features_have_header = True
    name = "Elliptic++"
