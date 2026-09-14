"""Regression tests for the lightweight temporal experiment examples."""

from pathlib import Path
from runpy import run_path

import pytest
import torch
from torch_geometric.data import Data, TemporalData

_EXAMPLES = Path(__file__).parents[1] / "examples"
JODIEStyleRiskModel = run_path(_EXAMPLES / "ibm_jodie_experiment.py")[
    "JODIEStyleRiskModel"
]
TGNStyleRiskModel = run_path(_EXAMPLES / "ibm_tgn_experiment.py")["TGNStyleRiskModel"]
window_loader = run_path(_EXAMPLES / "ibm_transaction_static_experiment.py")[
    "window_loader"
]


def test_window_loader_clips_targets_to_exact_split_boundaries() -> None:
    """A daily window may provide context but never cross a target split."""
    hour = 60 * 60 * 1_000_000_000
    graph = Data(
        x=torch.ones((4, 1)),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        node_y=torch.tensor([0, 1, 0, 1]),
        node_time=torch.tensor([0, 6 * hour, 12 * hour, 18 * hour]),
    )

    train = next(iter(window_loader(graph, 0, 12 * hour, batch_size=1)))
    validation = next(iter(window_loader(graph, 12 * hour, 24 * hour, batch_size=1)))

    assert torch.all(train.node_time[train.target_node_mask] < 12 * hour)
    assert torch.all(validation.node_time[validation.target_node_mask] >= 12 * hour)


@pytest.mark.parametrize("model_type", [JODIEStyleRiskModel, TGNStyleRiskModel])
def test_event_example_updates_are_trainable_and_keep_nanosecond_precision(
    model_type: type[torch.nn.Module],
) -> None:
    """State-update layers receive gradients and use precise time deltas."""
    model = model_type(num_accounts=2, message_dim=1, **_memory_size(model_type))
    base = 1_700_000_000_000_000_000
    model.last_time[0] = base
    event = TemporalData(
        src=torch.tensor([0]),
        dst=torch.tensor([1]),
        t=torch.tensor([base + 1_000_000_000]),
        msg=torch.tensor([[2.0]]),
        y=torch.tensor([1]),
    )

    elapsed = model._elapsed_hours(event)
    model(event).sum().backward()
    update_parameters = (
        model.update.parameters()
        if isinstance(model, JODIEStyleRiskModel)
        else list(model.message_function.parameters())
        + list(model.memory_updater.parameters())
    )

    assert elapsed.item() == pytest.approx(1 / 3600)
    assert all(parameter.grad is not None for parameter in update_parameters)


@pytest.mark.parametrize("model_type", [JODIEStyleRiskModel, TGNStyleRiskModel])
def test_event_example_same_timestamp_updates_are_order_invariant(
    model_type: type[torch.nn.Module],
) -> None:
    """Equal-time endpoint updates aggregate once per account deterministically."""
    kwargs = _memory_size(model_type)
    first = model_type(num_accounts=3, message_dim=1, **kwargs)
    second = model_type(num_accounts=3, message_dim=1, **kwargs)
    second.load_state_dict(first.state_dict())
    batch = TemporalData(
        src=torch.tensor([0, 1, 0]),
        dst=torch.tensor([1, 0, 2]),
        t=torch.tensor([10, 10, 10]),
        msg=torch.tensor([[1.0], [2.0], [3.0]]),
        y=torch.tensor([0, 1, 0]),
    )
    reversed_batch = TemporalData(
        src=batch.src.flip(0),
        dst=batch.dst.flip(0),
        t=batch.t.flip(0),
        msg=batch.msg.flip(0),
        y=batch.y.flip(0),
    )

    first.update_state(batch)
    second.update_state(reversed_batch)

    assert torch.equal(first.last_time, second.last_time)
    assert torch.allclose(first.memory, second.memory)


def _memory_size(model_type: type[torch.nn.Module]) -> dict[str, int]:
    """Use the example-specific name for the same compact state dimension."""
    if model_type is JODIEStyleRiskModel:
        return {"embedding_dim": 4}
    return {"memory_dim": 4}
