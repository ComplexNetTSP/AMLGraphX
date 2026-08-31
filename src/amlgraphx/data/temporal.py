"""Canonical temporal-axis helpers for datetime and ordinal event data.

AML datasets do not always provide wall-clock datetimes.  A discrete ``step``
can still define a rigorous ordering when its origin and duration are stated
explicitly.  This module maps that ordinal axis to a *logical* datetime while
retaining the raw step column, so existing temporal graph APIs can use one
typed representation without claiming that simulated steps are real dates.
"""

from datetime import UTC, datetime, timedelta

import polars as pl

DEFAULT_LOGICAL_TIME_ORIGIN = datetime(1970, 1, 1, tzinfo=UTC)


def logical_timestamp_from_step(
    frame: pl.LazyFrame,
    *,
    step_column: str,
    step_size: timedelta,
    timestamp_column: str = "timestamp",
    origin: datetime = DEFAULT_LOGICAL_TIME_ORIGIN,
) -> pl.LazyFrame:
    """Add a datetime timestamp derived from an ordinal step column.

    ``step`` remains untouched.  The generated timestamp is a logical axis:
    it preserves ordering and configured window widths, but does not imply a
    real-world calendar unless the caller supplies such an interpretation.

    Args:
        frame: Lazy table containing the ordinal time column.
        step_column: Integer-like column whose values identify time steps.
        step_size: Positive duration represented by one step.
        timestamp_column: Name for the generated canonical datetime column.
        origin: Datetime assigned to step zero.

    Returns:
        The input lazy frame with ``timestamp_column`` as ``Datetime[ns, UTC]``.

    Raises:
        TypeError: If the inputs do not describe a datetime-based logical axis.
        ValueError: If the column is absent or the step size is not positive.
    """
    if not isinstance(frame, pl.LazyFrame):
        raise TypeError("frame must be a polars.LazyFrame")
    if step_column not in frame.collect_schema().names():
        raise ValueError(f"Missing step column: {step_column}")
    if not isinstance(step_size, timedelta) or step_size <= timedelta(0):
        raise ValueError("step_size must be a positive datetime.timedelta")
    if not isinstance(origin, datetime):
        raise TypeError("origin must be a datetime")

    nanoseconds = _timedelta_to_nanoseconds(step_size)
    origin_ns = int(origin.timestamp() * 1_000_000_000)
    return frame.with_columns(
        pl.from_epoch(
            pl.lit(origin_ns) + pl.col(step_column).cast(pl.Int64) * nanoseconds,
            time_unit="ns",
        )
        .dt.replace_time_zone("UTC")
        .alias(timestamp_column)
    )


def _timedelta_to_nanoseconds(value: timedelta) -> int:
    """Return an exact integer nanosecond count for a Python timedelta."""
    return (
        value.days * 86_400_000_000_000
        + value.seconds * 1_000_000_000
        + value.microseconds * 1_000
    )


__all__ = ["DEFAULT_LOGICAL_TIME_ORIGIN", "logical_timestamp_from_step"]
