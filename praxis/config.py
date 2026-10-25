"""Paths and validated metadata shared by training and inference."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

CHECKPOINT_DIR = (
    Path(
        os.environ.get(
            "PRAXIS_CHECKPOINT_DIR", Path(__file__).resolve().parent.parent / "checkpoints"
        )
    )
    .expanduser()
    .resolve()
)
DEFAULT_STATS = CHECKPOINT_DIR / "normalization.npy"
DEFAULT_TASK_MAP = CHECKPOINT_DIR / "task_map.json"


def resolve_asset(value, default_dir=CHECKPOINT_DIR) -> Path:
    """Resolve a basename under the asset directory; preserve explicit paths."""
    raw = os.fspath(value)
    if not raw:
        raise ValueError("Asset path cannot be empty")
    path = Path(raw).expanduser()
    if raw.startswith(f".{os.sep}"):
        return path.absolute()
    explicit = path.is_absolute() or path.parent != Path(".")
    return path if explicit else Path(default_dir) / path


def load_stats(path=DEFAULT_STATS) -> tuple:
    """Read action/state means and positive standard deviations (7/7/8/8).

    The bundled legacy NumPy object array requires pickle support. Only load
    normalization files from a trusted source, as with checkpoint files.
    """
    source = resolve_asset(path)
    try:
        values = np.load(source, allow_pickle=True)
        if len(values) != 4:
            raise ValueError("expected four arrays: action mean/std and state mean/std")
        arrays = []
        for name, value, size in zip(
            ("action mean", "action std", "state mean", "state std"), values, (7, 7, 8, 8)
        ):
            array = np.asarray(value)
            if array.shape != (size,) or array.dtype.kind not in "fiu":
                raise ValueError(f"{name} must be a numeric array of shape ({size},)")
            if not np.isfinite(array).all():
                raise ValueError(f"{name} contains non-finite values")
            if name.endswith("std") and np.any(array <= 0):
                raise ValueError(f"{name} must be strictly positive")
            converted = array.astype(np.float32)
            if not np.isfinite(converted).all() or (
                name.endswith("std") and np.any(converted <= 0)
            ):
                raise ValueError(
                    f"{name} cannot be represented as finite float32 normalization values"
                )
            arrays.append(converted)
        return tuple(arrays)
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid normalization file {source}: {exc}") from exc


def load_task_map(path=DEFAULT_TASK_MAP) -> dict:
    """Load nonnegative task IDs mapped to (suite name, nonnegative task index)."""
    source = resolve_asset(path)
    try:
        with source.open(encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict) or not raw:
            raise ValueError("expected a nonempty JSON object")
        result = {}
        for key, value in raw.items():
            task = int(key)
            if task < 0 or task in result:
                raise ValueError(f"invalid or duplicate task ID {key!r}")
            if (
                not isinstance(value, list)
                or len(value) != 2
                or not isinstance(value[0], str)
                or not value[0].strip()
                or type(value[1]) is not int
                or value[1] < 0
            ):
                raise ValueError(
                    f"task {key!r} must contain [suite name, nonnegative integer index]"
                )
            result[task] = tuple(value)
        return result
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid task map {source}: {exc}") from exc
