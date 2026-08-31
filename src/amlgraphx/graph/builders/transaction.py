"""Transaction-as-node graph builder entry points.

The current builder creates temporal money-flow successor edges between
transaction nodes. Its implementation remains in ``graph.graphs`` for now;
this module establishes the intended stable import location without copying
the algorithm.

中文：
    当前 builder 在交易节点之间创建时间约束的资金流后继边。算法暂时保留在
    ``graph.graphs``，本模块只建立未来稳定的 import 位置，不复制实现。
"""

from ..graphs import TransactionGraph, build_transaction_graph

__all__ = ["TransactionGraph", "build_transaction_graph"]
