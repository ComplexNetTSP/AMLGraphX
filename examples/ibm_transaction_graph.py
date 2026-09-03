"""Prepare IBM HI-Small with explicit account and transaction semantics.

Run / 运行：
    ``uv run python examples/ibm_transaction_graph.py``

The example intentionally limits the number of rows so it is suitable for a
first API inspection. Increase ``MAX_TRANSACTIONS`` for a real experiment.
示例故意限制读取行数，方便第一次检查 API；真实实验时可以增大
``MAX_TRANSACTIONS``。

The static transaction-node example shows one complete graph plus
chronological node masks. Account-node graphs additionally support snapshots
and event streams because account identities persist over time:

    account nodes -> static / snapshots / event stream
    transaction nodes -> causal time-aware static graph

``edge_delta`` controls temporal-flow relations between transaction nodes.
Transaction attributes become node features and ``time_delta`` is a relation
edge feature. In account graphs, account metadata belongs to nodes and every
transaction remains an edge or event.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from amlgraphx.datasets import load_dataset
from amlgraphx.graph import GraphFeatureSpec, prepare_graph, prepare_pyg_graph
from amlgraphx.split import TemporalSplit, build_temporal_node_masks

MAX_TRANSACTIONS = 100_000
EDGE_DELTA = timedelta(hours=4)


def main(*, cache_dir: Path | None = None) -> None:
    """Load IBM HI-Small and inspect its supported graph representations."""
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

    # The high-level PyG facade maps transaction attributes and labels to nodes,
    # while the transaction-relation time delta becomes an edge feature.
    transaction_data = prepare_pyg_graph(
        transactions,
        node_type="transaction",
        temporal="static",
        edge_delta=EDGE_DELTA,
        features=GraphFeatureSpec(
            node_columns=("amount",),
            edge_columns=("time_delta",),
            label_column="label",
        ),
    )
    print("Transaction PyG data / 交易节点 PyG 数据:", transaction_data)

    # Account-as-node event streams need no manual graph conversion: amount is
    # the event message and the transaction label is the event target.
    event_stream = prepare_pyg_graph(
        transactions,
        node_type="account",
        temporal="event_stream",
        features=GraphFeatureSpec(
            edge_columns=("amount",),
            label_column="label",
        ),
    )
    print(
        "Account event stream / 账户事件流: "
        f"{event_stream.num_nodes} nodes, {event_stream.num_events} events"
    )
    print("First event messages / 前几个事件特征:")
    print(event_stream.msg[:3])


if __name__ == "__main__":
    main()
