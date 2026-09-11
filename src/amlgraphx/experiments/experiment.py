"""A lightweight experiment lifecycle for AMLGraphX model inputs.

Researchers build datasets, graph representations, and loaders with the
explicit AMLGraphX APIs. ``Experiment`` then checks their model against a
small structural dummy batch, fits only on training data, predicts on held-out
test data, and evaluates one aligned transaction-risk score per target.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import wraps
from typing import Any, Literal

import numpy as np
import torch
from pytorch_lightning import Trainer
from torch import Tensor, nn
from torch_geometric.data import Data, TemporalData
from torchmetrics import Metric

from amlgraphx.data import SnapshotBatch
from amlgraphx.evaluation import BinaryRiskMetrics, evaluate_binary_risk_scores
from amlgraphx.training import (
    EventStreamBinaryPredictor,
    ModelContractError,
    SnapshotBinaryEdgePredictor,
    SnapshotBinaryNodePredictor,
    StaticBinaryEdgePredictor,
    StaticBinaryNodePredictor,
)

Representation = Literal["static", "snapshot", "event_stream"]
TargetKind = Literal["node", "edge", "event"]
PredictionKind = TargetKind | Literal["row"]


@dataclass(frozen=True, slots=True)
class BinaryRiskTask:
    """Describe where a binary AML/fraud label lives in a model input.

    ``target_mask_attr`` is normally a chronological split mask for full-graph
    execution or a ``target_*_mask`` emitted by a bounded temporal loader.
    """

    representation: Representation
    target_kind: PredictionKind
    label_attr: str | None = None
    target_mask_attr: str | None = None
    train_mask_attr: str | None = None
    validation_mask_attr: str | None = None
    test_mask_attr: str | None = None

    def __post_init__(self) -> None:
        allowed = {
            "static": {"node", "edge"},
            "snapshot": {"node", "edge"},
            "event_stream": {"event"},
        }
        if self.representation not in allowed:
            raise ValueError(f"unknown representation {self.representation!r}")
        if self.target_kind not in allowed[self.representation]:
            raise ValueError(
                f"{self.representation!r} does not support {self.target_kind!r} targets"
            )
        if self.label_attr is None:
            defaults = {"node": "node_y", "edge": "edge_y", "event": "y"}
            object.__setattr__(self, "label_attr", defaults[self.target_kind])

    def mask_attr(self, stage: Literal["train", "validation", "test"]) -> str | None:
        """Return a stage override, a shared window mask, or the static default."""
        explicit = getattr(self, f"{stage}_mask_attr")
        if explicit is not None:
            return explicit
        if self.target_mask_attr is not None:
            return self.target_mask_attr
        if self.representation != "static":
            return None
        suffix = "" if self.target_kind == "node" else "_edge"
        return f"{stage}{suffix}_mask"


@dataclass(frozen=True, slots=True)
class RiskPredictions:
    """Held-out binary labels and risk scores aligned to one prediction unit."""

    labels: Tensor
    scores: Tensor
    target_kind: TargetKind


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Return fitted Lightning state, composable metrics, and AML score metrics."""

    predictor: nn.Module
    test_metrics: Mapping[str, float]
    predictions: RiskPredictions
    risk_metrics: BinaryRiskMetrics | None


@dataclass(frozen=True, slots=True)
class TabularExperimentResult:
    """Return a fitted classical estimator and its frozen test risk scores."""

    model: Any
    metrics: Mapping[str, float]
    predictions: RiskPredictions
    risk_metrics: BinaryRiskMetrics | None


def _validate_before_fit(method: Any) -> Any:
    """Run the researcher-model contract check before any optimiser step."""

    @wraps(method)
    def wrapped(
        self: Experiment, train_dataloaders: Iterable[Any], *args: Any, **kwargs: Any
    ) -> Any:
        self.validate_model(train_dataloaders)
        return method(self, train_dataloaders, *args, **kwargs)

    return wrapped


class Experiment:
    """Fit a researcher-defined binary risk model with one concise lifecycle.

    The class deliberately does not load a dataset or invent a graph. A
    researcher first uses ``datasets``, ``graph``, ``split``, and ``data`` to
    make explicit inputs, then passes their train/validation/test loaders here.
    The selected task chooses the matching AMLGraphX predictor contract.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        task: BinaryRiskTask,
        metrics: Mapping[str, Metric] | None = None,
        loss: nn.Module | None = None,
        predictor_kwargs: Mapping[str, Any] | None = None,
        trainer_kwargs: Mapping[str, Any] | None = None,
        evaluation_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """Store a model and create its representation-specific predictor."""
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if metrics is not None and any(
            not isinstance(metric, Metric) for metric in metrics.values()
        ):
            raise TypeError("metrics must contain torchmetrics.Metric instances")
        self.task = task
        self.loss = loss or nn.BCEWithLogitsLoss()
        self.metrics = dict(metrics or {})
        self.predictor = _make_predictor(
            model, task, self.loss, self.metrics, predictor_kwargs
        )
        options = dict(trainer_kwargs or {})
        options.setdefault("enable_progress_bar", True)
        options.setdefault("enable_model_summary", False)
        # Tracking/checkpoint policies belong to explicit future integrations.
        options.setdefault("logger", False)
        options.setdefault("enable_checkpointing", False)
        self.trainer = Trainer(**options)
        self.evaluation_kwargs = dict(evaluation_kwargs or {})

    @_validate_before_fit
    def fit(
        self,
        train_dataloaders: Iterable[Any],
        validation_dataloaders: Iterable[Any] | None = None,
    ) -> Experiment:
        """Fit only on train loaders and use validation loaders for monitoring."""
        self.trainer.fit(
            self.predictor,
            train_dataloaders=train_dataloaders,
            val_dataloaders=validation_dataloaders,
        )
        return self

    def validate_model(self, train_dataloaders: Iterable[Any]) -> None:
        """Check model input/output on a compact dummy derived from one batch."""
        try:
            batch = next(iter(train_dataloaders))
        except StopIteration as error:
            raise ValueError(
                "train_dataloaders must yield at least one batch"
            ) from error
        dummy = _dummy_batch(batch)
        was_training = self.predictor.training
        self.predictor.eval()
        try:
            with torch.no_grad():
                self.predictor.forward(dummy)
        except (ModelContractError, RuntimeError, TypeError, AttributeError) as error:
            raise ModelContractError(
                "researcher model does not satisfy the AMLGraphX input/output contract"
            ) from error
        finally:
            self.predictor.train(was_training)

    def run(
        self,
        *,
        train_dataloaders: Iterable[Any],
        validation_dataloaders: Iterable[Any] | None,
        test_dataloaders: Iterable[Any],
    ) -> ExperimentResult:
        """Fit, test, predict, and evaluate one frozen held-out test sequence."""
        self.fit(train_dataloaders, validation_dataloaders)
        test_result = self.trainer.test(self.predictor, dataloaders=test_dataloaders)
        prediction_batches = _predict_batches(self.predictor, test_dataloaders)
        predictions = _collect_predictions(
            test_dataloaders, prediction_batches, self.task
        )
        risk_metrics = _evaluate_predictions(predictions, self.evaluation_kwargs)
        metrics = test_result[0] if test_result else {}
        return ExperimentResult(self.predictor, metrics, predictions, risk_metrics)


class TabularExperiment:
    """Run a sklearn-style binary estimator without adding a second workflow API.

    The estimator must expose ``fit`` and one of ``predict_proba``,
    ``decision_function``, or ``predict``. Training uses only the supplied
    train arrays; validation data deliberately remains a researcher-controlled
    estimator concern, just as model architecture does for graph experiments.
    """

    def __init__(
        self,
        model: Any,
        *,
        metrics: Mapping[str, Any] | None = None,
        evaluation_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """Store and validate a sklearn-compatible binary-risk estimator."""
        if not callable(getattr(model, "fit", None)):
            raise TypeError("tabular model must define fit(X, y)")
        if not any(
            callable(getattr(model, name, None))
            for name in ("predict_proba", "decision_function", "predict")
        ):
            raise TypeError(
                "tabular model must define predict_proba, decision_function, or predict"
            )
        if metrics is not None and any(
            not callable(metric) for metric in metrics.values()
        ):
            raise TypeError(
                "tabular metrics must be callables accepting (labels, scores)"
            )
        self.model = model
        self.metrics = dict(metrics or {})
        self.evaluation_kwargs = dict(evaluation_kwargs or {})

    def run(
        self,
        train_features: Any,
        train_labels: Any,
        test_features: Any,
        test_labels: Any,
    ) -> TabularExperimentResult:
        """Fit on train arrays only, then score and evaluate the frozen test split."""
        self.model.fit(train_features, train_labels)
        scores = _tabular_scores(self.model, test_features)
        labels = torch.as_tensor(np.asarray(test_labels), dtype=torch.long).reshape(-1)
        predictions = RiskPredictions(labels, torch.from_numpy(scores), "row")
        metric_values = {
            name: float(metric(labels.numpy(), scores))
            for name, metric in self.metrics.items()
        }
        return TabularExperimentResult(
            self.model,
            metric_values,
            predictions,
            _evaluate_predictions(predictions, self.evaluation_kwargs),
        )


def _make_predictor(
    model: nn.Module,
    task: BinaryRiskTask,
    loss: nn.Module,
    metrics: Mapping[str, Metric],
    predictor_kwargs: Mapping[str, Any] | None,
) -> nn.Module:
    """Select the existing predictor whose output axis matches the task target."""
    kwargs = dict(predictor_kwargs or {})
    if task.representation == "static":
        predictor_type = (
            StaticBinaryNodePredictor
            if task.target_kind == "node"
            else StaticBinaryEdgePredictor
        )
        kwargs.setdefault("train_mask_attr", task.mask_attr("train"))
        kwargs.setdefault("validation_mask_attr", task.mask_attr("validation"))
        kwargs.setdefault("test_mask_attr", task.mask_attr("test"))
    elif task.representation == "snapshot":
        predictor_type = (
            SnapshotBinaryNodePredictor
            if task.target_kind == "node"
            else SnapshotBinaryEdgePredictor
        )
        if task.target_mask_attr is not None:
            kwargs.setdefault("target_mask_attr", task.target_mask_attr)
    else:
        predictor_type = EventStreamBinaryPredictor
        if task.target_mask_attr is not None:
            kwargs.setdefault("event_mask_attr", task.target_mask_attr)
    kwargs.setdefault("target_attr", task.label_attr)
    return predictor_type(model, loss, metrics=metrics, **kwargs)


def _collect_predictions(
    dataloader: Iterable[Any], prediction_batches: Sequence[Any], task: BinaryRiskTask
) -> RiskPredictions:
    """Select held-out labels and scores with the same target mask on each batch."""
    labels: list[Tensor] = []
    scores: list[Tensor] = []
    for batch, predicted in zip(dataloader, prediction_batches, strict=True):
        target = _target_graph(batch, task)
        label = getattr(target, task.label_attr)
        score = predicted.reshape(-1)
        mask = _target_mask(target, task.mask_attr("test"), label.numel())
        labels.append(label[mask].detach().cpu().to(dtype=torch.long))
        scores.append(score[mask].detach().cpu().to(dtype=torch.float32))
    if not labels:
        raise ValueError("test_dataloaders must yield at least one batch")
    return RiskPredictions(torch.cat(labels), torch.cat(scores), task.target_kind)


def _predict_batches(predictor: nn.Module, dataloader: Iterable[Any]) -> list[Tensor]:
    """Predict directly so TemporalDataLoader event counts stay aligned to scores.

    Lightning's generic prediction result handling treats some temporal batches
    as collections. Calling the documented predictor hook directly preserves
    one tensor per input batch while retaining its state-reset/update contract.
    """
    start = getattr(predictor, "on_predict_epoch_start", None)
    if callable(start):
        start()
    device = _module_device(predictor)
    was_training = predictor.training
    predictor.eval()
    outputs: list[Tensor] = []
    try:
        with torch.no_grad():
            for index, batch in enumerate(dataloader):
                moved = batch.to(device) if hasattr(batch, "to") else batch
                scores = predictor.predict_step(moved, index)
                outputs.append(scores.detach().cpu())
    finally:
        predictor.train(was_training)
    return outputs


def _module_device(module: nn.Module) -> torch.device:
    """Return a parameter or buffer device without requiring trainable weights."""
    value = next(module.parameters(), None)
    if value is None:
        value = next(module.buffers(), None)
    return value.device if value is not None else torch.device("cpu")


def _evaluate_predictions(
    predictions: RiskPredictions, options: Mapping[str, Any]
) -> BinaryRiskMetrics | None:
    """Evaluate a complete test split when it contains both binary classes."""
    if torch.unique(predictions.labels).numel() < 2:
        return None
    return evaluate_binary_risk_scores(
        predictions.labels.numpy(), predictions.scores.numpy(), **options
    )


def _tabular_scores(model: Any, features: Any) -> np.ndarray:
    """Convert common sklearn binary-estimator outputs into one risk-score vector."""
    if callable(getattr(model, "predict_proba", None)):
        probability = np.asarray(model.predict_proba(features))
        if probability.ndim != 2 or probability.shape[1] < 2:
            raise ValueError(
                "predict_proba must return an array with a positive-class column"
            )
        return probability[:, 1].astype(np.float32, copy=False)
    if callable(getattr(model, "decision_function", None)):
        return np.asarray(model.decision_function(features), dtype=np.float32).reshape(
            -1
        )
    return np.asarray(model.predict(features), dtype=np.float32).reshape(-1)


def _target_graph(batch: Any, task: BinaryRiskTask) -> Any:
    """Use the target graph inside a snapshot batch and the batch otherwise."""
    return batch.target if task.representation == "snapshot" else batch


def _target_mask(batch: Any, name: str | None, count: int) -> Tensor:
    """Read an optional boolean target selector, defaulting to every item."""
    if name is None:
        return torch.ones(count, dtype=torch.bool, device=_batch_device(batch))
    value = getattr(batch, name, None)
    if (
        not isinstance(value, Tensor)
        or value.dtype != torch.bool
        or value.numel() != count
    ):
        raise ModelContractError(
            f"{name} must be a boolean tensor aligned to prediction targets"
        )
    return value


def _batch_device(batch: Any) -> torch.device:
    """Choose the target device without assuming that x exists."""
    for name in ("x", "node_y", "edge_y", "y", "t"):
        value = getattr(batch, name, None)
        if isinstance(value, Tensor):
            return value.device
    return torch.device("cpu")


def _dummy_batch(batch: Any) -> Any:
    """Create a tiny structural input so contract checks do not use real scale."""
    if isinstance(batch, SnapshotBatch):
        return SnapshotBatch(
            context=tuple(_dummy_data(graph) for graph in batch.context),
            target=_dummy_data(batch.target),
        )
    if isinstance(batch, TemporalData):
        return _dummy_events(batch)
    if isinstance(batch, Data):
        return _dummy_data(batch)
    raise TypeError(
        "AMLGraphX Experiment accepts PyG Data, TemporalData, or SnapshotBatch"
    )


def _dummy_data(batch: Data) -> Data:
    """Keep at most four nodes and one synthetic edge with aligned attributes."""
    node_count = min(int(batch.num_nodes or 1), 4)
    edge_count = 1 if int(batch.num_edges) else 0
    edge_label = getattr(batch, "edge_label", None)
    edge_target_count = (
        min(edge_label.numel(), 1) if isinstance(edge_label, Tensor) else edge_count
    )
    dummy = Data(num_nodes=node_count)
    for name, value in batch.to_dict().items():
        if name == "edge_index":
            dummy.edge_index = torch.zeros((2, edge_count), dtype=torch.long)
        elif name == "edge_label_index":
            dummy.edge_label_index = torch.zeros(
                (2, edge_target_count), dtype=torch.long
            )
        elif name in {
            "edge_label",
            "edge_label_time",
            "target_edge_mask",
            "target_edge_time",
        }:
            if isinstance(value, Tensor):
                setattr(dummy, name, value[:edge_target_count].clone())
        elif (
            isinstance(value, Tensor)
            and value.ndim > 0
            and value.shape[0] == int(batch.num_nodes or 0)
        ):
            setattr(dummy, name, value[:node_count].clone())
        elif (
            isinstance(value, Tensor)
            and value.ndim > 0
            and value.shape[0] == int(batch.num_edges)
        ):
            setattr(dummy, name, value[:edge_count].clone())
        elif isinstance(value, Tensor):
            setattr(dummy, name, value.clone())
        elif name not in {"num_nodes", "num_edges"}:
            setattr(dummy, name, value)
    for name in ("train_mask", "validation_mask", "test_mask", "target_node_mask"):
        if hasattr(dummy, name):
            setattr(dummy, name, torch.ones(node_count, dtype=torch.bool))
    for name in (
        "train_edge_mask",
        "validation_edge_mask",
        "test_edge_mask",
        "target_edge_mask",
    ):
        if hasattr(dummy, name):
            count = edge_target_count if name == "target_edge_mask" else edge_count
            setattr(dummy, name, torch.ones(count, dtype=torch.bool))
    return dummy


def _dummy_events(batch: TemporalData) -> TemporalData:
    """Create one remapped interaction with the original message dimensions."""
    message = batch.msg[:1].clone() if batch.msg is not None else torch.empty((1, 0))
    kwargs: dict[str, Tensor] = {}
    if getattr(batch, "y", None) is not None:
        kwargs["y"] = batch.y[:1].clone()
    dummy = TemporalData(
        src=torch.tensor([0]),
        dst=torch.tensor([1]),
        t=torch.zeros(1, dtype=batch.t.dtype),
        msg=message,
        **kwargs,
    )
    if getattr(batch, "x", None) is not None:
        dummy.x = batch.x[: min(2, batch.x.shape[0])].clone()
    for name in ("event_mask", "target_event_mask"):
        if hasattr(batch, name):
            setattr(dummy, name, torch.ones(1, dtype=torch.bool))
    return dummy


__all__ = [
    "BinaryRiskTask",
    "Experiment",
    "ExperimentResult",
    "RiskPredictions",
    "TabularExperiment",
    "TabularExperimentResult",
]
