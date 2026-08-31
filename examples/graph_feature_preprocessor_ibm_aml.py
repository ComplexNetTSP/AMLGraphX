"""Download IBM AML data, load a bounded stream, and enrich it with GFP.

Run / 运行：
    uv run python examples/graph_feature_preprocessor_ibm_aml.py

The first run downloads the public ``hi-small`` archive into AMLGraphX's
standard cache. Change ``MAX_TRANSACTIONS`` before using a larger experiment.
首次运行会把公开的 ``hi-small`` 数据下载到 AMLGraphX 的标准缓存目录；在运行
更大实验前请调整 ``MAX_TRANSACTIONS``。
"""

from __future__ import annotations

from itertools import batched
from os import cpu_count

import numpy as np
import polars as pl

from amlgraphx.datasets import IBMAML
from amlgraphx.tabular import GraphFeaturePreprocessor

MAX_TRANSACTIONS = 10_000
WARMUP_TRANSACTIONS = 2_048
BATCH_SIZE = 128


def load_gfp_matrix(dataset: IBMAML, limit: int) -> np.ndarray:
    """Load an ordered, numeric GFP matrix from canonical AMLGraphX transactions.

    GFP requires ``[edge_id, source_id, target_id, timestamp, ...numeric]``.
    GFP 的输入必须是 ``[edge_id, source_id, target_id, timestamp, ...数值列]``。
    """
    transactions = (
        dataset.transactions()
        .select("transaction_id", "source", "target", "timestamp", "amount", "label")
        .drop_nulls(["timestamp", "amount"])
        .sort("timestamp", "transaction_id")
        .limit(limit)
        .collect()
    )
    if transactions.height <= WARMUP_TRANSACTIONS:
        raise ValueError(
            "Increase MAX_TRANSACTIONS so the stream has a warm-up and a batch"
        )

    # Give every account one stable integer ID across both source and target columns.
    # 为来源和去向账户建立同一份稳定的整数 ID 映射。
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

    # Column 4 is amount, so it is the value used by vertex statistics below.
    # 第 4 列是金额，下面的账户统计会使用它。
    return encoded.select(
        pl.col("edge_id").cast(pl.Float64),
        pl.col("source_id").cast(pl.Float64),
        pl.col("target_id").cast(pl.Float64),
        pl.col("timestamp").dt.epoch(time_unit="s").cast(pl.Float64),
        pl.col("amount").cast(pl.Float64),
    ).to_numpy()


def main() -> None:
    # download() is explicit; transactions() then lazily loads the canonical schema.
    # download() 显式下载数据；transactions() 随后惰性读取统一交易 schema。
    dataset = IBMAML("hi-small")
    print(f"Dataset directory / 数据目录: {dataset.download()}")
    features = load_gfp_matrix(dataset, MAX_TRANSACTIONS)

    gfp = GraphFeaturePreprocessor()
    one_day = 24 * 60 * 60
    gfp.set_params(
        {
            "num_threads": min(12, cpu_count() or 1),
            "time_window": one_day,
            "vertex_stats": True,
            "vertex_stats_cols": [4],  # amount / 金额
            "fan": False,
            "degree": False,
            "scatter-gather": True,
            "temp-cycle": True,
            "lc-cycle": True,
            "lc-cycle_len": 10,
            "vertex_stats_tw": one_day,
            "scatter-gather_tw": 6 * 60 * 60,
            "temp-cycle_tw": one_day,
            "lc-cycle_tw": one_day,
        }
    )

    # fit() loads history without producing features; it is the streaming warm-up.
    # fit() 只载入历史交易而不输出特征，用作流式处理的预热阶段。
    gfp.fit(features[:WARMUP_TRANSACTIONS])

    enriched_batches = []
    for batch in batched(features[WARMUP_TRANSACTIONS:], BATCH_SIZE):
        batch_array = np.asarray(batch, dtype=np.float64)
        # transform() follows GFP/SnapML batch semantics and uses the Rust Rayon pool.
        # transform() 使用 GFP/SnapML 的 batch 语义，并调用 Rust Rayon 线程池。
        enriched_batches.append(gfp.transform(batch_array))
    enriched = np.vstack(enriched_batches)

    print(f"Raw matrix / 原始矩阵: {features.shape}")
    print(f"GFP matrix / GFP 特征矩阵: {enriched.shape}")
    print(f"Active graph edges / 内存图中的活跃边: {gfp.active_edge_count}")
    # For strict event-by-event causal evaluation, replace a batch call above with
    # gfp.transform_causal(batch_array). It is intentionally slower but excludes
    # visibility of later rows in the same batch.
    # 如需严格逐事件因果评估，将上面的 batch 调用替换为 transform_causal；它会更慢，
    # 但不会让同一 batch 中较早交易看到后来的交易。


if __name__ == "__main__":
    main()
