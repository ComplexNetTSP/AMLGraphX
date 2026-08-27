"""Account-as-node graph builder entry points.

The implementation currently remains in ``amlgraphx.graph.graphs`` so the
existing public path stays compatible. This module is the canonical semantic
home for the account builder during the staged migration.

中文：
    当前实现暂时保留在 ``amlgraphx.graph.graphs``，以维持已有公共路径兼容。
    本模块是迁移期间账户为节点 builder 的规范语义入口。
"""

from ..graphs import AccountGraph, build_account_graph

__all__ = ["AccountGraph", "build_account_graph"]
