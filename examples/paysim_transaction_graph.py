"""Prepare IBM HI-Small as transaction-node static and snapshot graphs.

Run / 运行：
    ``uv run python examples/paysim_transaction_graph.py``

The example intentionally limits the number of rows so it is suitable for a
first API inspection. Increase ``MAX_TRANSACTIONS`` for a real experiment.
示例故意限制读取行数，方便第一次检查 API；真实实验时可以增大
``MAX_TRANSACTIONS``。

The static example also shows the research split protocol used by the paper:
one complete graph plus chronological node masks. The two graph modes share
the same canonical transaction table:

    dataset -> canonical transactions -> transaction graph -> masks/snapshots

``edge_delta`` controls temporal-flow edges between transaction nodes. The
snapshot ``bin_size`` and ``stride`` control how the already-defined graph is
materialized over time; they are different concepts.
"""

from __future__ import annotations

from datetime import timedelta
from itertools import islice
from pathlib import Path

from amlgraphx.datasets import load_dataset
from amlgraphx.graph import prepare_graph
from amlgraphx.split import TemporalSplit, build_temporal_node_masks

MAX_TRANSACTIONS = 100_000
EDGE_DELTA = timedelta(hours=4)
SNAPSHOT_BIN = timedelta(days=1)
SNAPSHOT_STRIDE = timedelta(days=1)


def main(*, cache_dir: Path | None = None) -> None:
    """Load IBM HI-Small and inspect transaction-node graph representations."""
    # ``load_dataset`` downloads and prepares the adapter; the transaction
    # table itself remains lazy until a graph builder needs to materialize it.
    # ``load_dataset`` 会下载并准备 adapter；交易表在 builder 需要时才物化。
    dataset = load_dataset("ibm-aml", variant="hi-small", cache_dir=cache_dir)
    transactions = dataset.transactions().limit(MAX_TRANSACTIONS)

    print("Canonical transaction schema / 统一交易 schema:")
    print(transactions.collect_schema())

    # Mode 1: one time-aware static graph with transactions as nodes.
    # 模式 1：一张时间感知静态图，交易本身是节点。
    static_graph = prepare_graph(
        transactions,
        node_type="transaction",
        temporal="static",
        edge_delta=EDGE_DELTA,
    )
    print(
        "Static transaction graph / 时间感知静态交易图: "
        f"{static_graph.num_nodes} nodes, {static_graph.num_edges} edges"
    )

    # Keep the complete graph and create chronological node masks. This is a
    # transductive temporal protocol: masks control labels/loss, not edges.
    # 保留完整图并创建时间节点 mask；mask 控制标签/loss，而不是删除边。
    timestamp_start = static_graph.nodes["timestamp"].min()
    timestamp_end = static_graph.nodes["timestamp"].max()
    time_span = timestamp_end - timestamp_start
    masks = build_temporal_node_masks(
        static_graph,
        TemporalSplit(
            train_end=timestamp_start + time_span * 0.6,
            validation_end=timestamp_start + time_span * 0.8,
        ),
    )
    print(
        "Chronological masks / 时间节点 mask: "
        f"train={masks.train_mask.sum().item()}, "
        f"validation={masks.validation_mask.sum().item()}, "
        f"test={masks.test_mask.sum().item()}"
    )

    # Mode 2: a sequence of daily transaction-node snapshots.
    # 模式 2：按天划分的 transaction-node snapshot 序列。
    snapshot_iterator = prepare_graph(
        transactions,
        node_type="transaction",
        temporal="snapshot",
        edge_delta=EDGE_DELTA,
        bin_size=SNAPSHOT_BIN,
        stride=SNAPSHOT_STRIDE,
        drop_last=False,
    )
    print("First snapshots / 前几个 snapshot:")
    for snapshot in islice(snapshot_iterator, 3):
        print(
            f"  #{snapshot.index}: "
            f"[{snapshot.start_time}, {snapshot.end_time}) -> "
            f"{snapshot.num_nodes} nodes, {snapshot.num_edges} edges, "
            f"edge_index={tuple(snapshot.edge_index.shape)}"
        )


if __name__ == "__main__":
    main()
