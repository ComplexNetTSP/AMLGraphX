"""Evaluate XGBoost on chronological IBM AML transactions enriched by GFP.

Run / 运行：
    uv run python examples/ibm_hi_small_gfp_xgboost.py

The example keeps a tabular transaction representation: GraphFeaturePreprocessor
(GFP) derives graph features, then XGBoost ranks later transactions by laundering
risk. It uses a temporary Hugging Face cache and extraction directory; all
downloaded IBM HI-Small data is deleted when the process leaves the context.

For a strict event-by-event causal protocol, replace the batch calls to
``transform`` with ``transform_causal`` after defining an explicit policy for
equal timestamps. The batch mode below is SnapML-compatible but allows rows in
the same batch to see one another's graph state.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import polars as pl
import torch

from amlgraphx.baselines import XGBoostBaseline
from amlgraphx.datasets import IBMAML
from amlgraphx.evaluation import (
    F1,
    AveragePrecision,
    F1AtK,
    LiftAtK,
    Precision,
    PrecisionAtK,
    Recall,
    RecallAtK,
    RocAuc,
)
from amlgraphx.tabular import GraphFeaturePreprocessor

MAX_TRANSACTIONS = 75_000
WARMUP_TRANSACTIONS = 1_024
BATCH_SIZE = 128
RANDOM_SEED = 42


def load_ordered_transactions(
    dataset: IBMAML, limit: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return ordered GFP inputs and laundering labels from canonical rows.

    The feature columns are ``edge_id, source_id, target_id, timestamp, amount``.
    ``timestamp`` adds a deterministic microsecond tie-breaker only to order
    transactions with equal published timestamps; the one-day GFP windows are
    unaffected for the bounded example stream.
    """
    transactions = (
        dataset.transactions()
        .select("transaction_id", "source", "target", "timestamp", "amount", "label")
        .drop_nulls(["source", "target", "timestamp", "amount", "label"])
        .with_columns(pl.col("label").cast(pl.Int8))
        .filter(pl.col("label").is_in((0, 1)))
        .sort("timestamp", "transaction_id")
        .limit(limit)
        .collect()
    )
    if transactions.height <= WARMUP_TRANSACTIONS + 3:
        raise ValueError("Increase MAX_TRANSACTIONS to leave warm-up and split targets")

    accounts = (
        pl.concat(
            [
                transactions.select(pl.col("source").alias("account")),
                transactions.select(pl.col("target").alias("account")),
            ]
        )
        .unique()
        .sort("account")
        .with_row_index("account_id")
    )
    encoded = (
        transactions.with_row_index("edge_id")
        .join(accounts, left_on="source", right_on="account")
        .rename({"account_id": "source_id"})
        .join(accounts, left_on="target", right_on="account")
        .rename({"account_id": "target_id"})
    )
    feature_frame = encoded.select(
        pl.col("edge_id").cast(pl.Float64),
        pl.col("source_id").cast(pl.Float64),
        pl.col("target_id").cast(pl.Float64),
        (
            pl.col("timestamp").dt.epoch(time_unit="us").cast(pl.Float64) / 1_000_000
            + pl.col("edge_id").cast(pl.Float64) / 1_000_000
        ).alias("timestamp"),
        pl.col("amount").cast(pl.Float64),
    )
    return feature_frame.to_numpy(), encoded["label"].to_numpy().astype(np.int8)


def enrich_in_batches(
    gfp: GraphFeaturePreprocessor, features: np.ndarray
) -> np.ndarray:
    """Enrich ordered transactions with GFP's documented batch semantics."""
    batches = [
        gfp.transform(features[start : start + BATCH_SIZE])
        for start in range(0, features.shape[0], BATCH_SIZE)
    ]
    return np.vstack(batches)


def require_both_classes(y: np.ndarray, name: str) -> None:
    """Fail clearly when a small chronological slice lacks a class."""
    if np.unique(y).size != 2:
        raise ValueError(f"{name} split has one class; increase MAX_TRANSACTIONS")


def build_test_metrics(sample_count: int) -> dict[str, torch.nn.Module]:
    """Instantiate the explicit threshold and investigation metrics to report."""
    top_0_1_percent = max(1, int(np.ceil(sample_count * 0.001)))
    top_1_percent = max(1, int(np.ceil(sample_count * 0.01)))
    return {
        "precision": Precision(threshold=0.5),
        "recall": Recall(threshold=0.5),
        "f1": F1(threshold=0.5),
        "average_precision": AveragePrecision(),
        "roc_auc": RocAuc(),
        "precision_at_0_1_percent": PrecisionAtK(top_0_1_percent),
        "recall_at_0_1_percent": RecallAtK(top_0_1_percent),
        "f1_at_0_1_percent": F1AtK(top_0_1_percent),
        "lift_at_0_1_percent": LiftAtK(top_0_1_percent),
        "precision_at_1_percent": PrecisionAtK(top_1_percent),
        "recall_at_1_percent": RecallAtK(top_1_percent),
        "f1_at_1_percent": F1AtK(top_1_percent),
        "lift_at_1_percent": LiftAtK(top_1_percent),
    }


def main() -> None:
    """Download temporary data, train a tabular baseline, and print test metrics."""
    with TemporaryDirectory(prefix="amlgraphx-ibm-gfp-xgboost-") as temporary_root:
        root = Path(temporary_root)
        dataset = IBMAML(
            "hi-small",
            cache_dir=root / "hf-cache",
            local_dir=root / "data",
        )
        print(f"Temporary dataset directory / 临时数据目录: {dataset.download()}")
        raw_features, labels = load_ordered_transactions(dataset, MAX_TRANSACTIONS)

        gfp = GraphFeaturePreprocessor()
        one_day = 24 * 60 * 60
        gfp.set_params(
            {
                "num_threads": 1,
                "time_window": one_day,
                "vertex_stats_cols": [4],
                "fan": False,
                "degree": False,
                "scatter-gather": True,
                "temp-cycle": True,
                "lc-cycle": True,
                "vertex_stats_tw": one_day,
                "scatter-gather_tw": 6 * 60 * 60,
                "temp-cycle_tw": one_day,
                "lc-cycle_tw": one_day,
            }
        )
        gfp.fit(raw_features[:WARMUP_TRANSACTIONS])
        enriched = enrich_in_batches(gfp, raw_features[WARMUP_TRANSACTIONS:])
        target_labels = labels[WARMUP_TRANSACTIONS:]
        if not np.isfinite(enriched).all():
            raise ValueError("GFP produced non-finite features")

        # Account/edge IDs are graph identifiers, not tabular model features.
        # GFP output retains the raw layout, so only amount and appended features remain.
        X = enriched[:, 4:].astype(np.float32)
        train_end = int(0.75 * X.shape[0])
        X_train, X_test = X[:train_end], X[train_end:]
        y_train, y_test = target_labels[:train_end], target_labels[train_end:]
        for name, y in (("train", y_train), ("test", y_test)):
            require_both_classes(y, name)

        scale_pos_weight = float(
            np.count_nonzero(y_train == 0) / np.count_nonzero(y_train)
        )
        model = XGBoostBaseline(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
            n_jobs=1,
            random_state=RANDOM_SEED,
        ).fit(X_train, y_train)

        # Every class owns its state. ``update`` can be called for every model
        # batch, then ``compute`` returns the complete chronological test result.
        # ``0.5`` is predeclared here; production should select a threshold on
        # validation data and freeze it before observing test labels.
        risk_score = model.predict_proba(X_test)[:, 1]
        metrics = build_test_metrics(y_test.size)
        score_tensor = torch.from_numpy(risk_score.astype(np.float32))
        target_tensor = torch.from_numpy(y_test)
        for start in range(0, y_test.size, BATCH_SIZE):
            for metric in metrics.values():
                metric.update(
                    score_tensor[start : start + BATCH_SIZE],
                    target_tensor[start : start + BATCH_SIZE],
                )
        results = {name: metric.compute().item() for name, metric in metrics.items()}
        print(f"Rows / 交易数: raw={raw_features.shape[0]}, targets={X.shape[0]}")
        print(f"GFP matrix / GFP 特征矩阵: {enriched.shape}")
        print("Test metrics / 测试指标:")
        for name, value in results.items():
            print(f"  {name}: {value}")

    print(f"Temporary data deleted / 临时数据已删除: {not root.exists()}")


if __name__ == "__main__":
    main()
