"""Shared paths and serialization helpers for local input preparation."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "notebooks" / "julia" / "data" / "preparation_records"
REPORT.mkdir(parents=True, exist_ok=True)

_risklabai_specification = importlib.util.find_spec("RiskLabAI")
PYTHON = (
    Path(_risklabai_specification.origin).resolve().parent
    if _risklabai_specification and _risklabai_specification.origin
    else ROOT / ".missing-risklabai"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_exact(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Existing generated file differs: {path}")
    else:
        path.write_bytes(content)


def save_metadata(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalize = lambda value: b"\n".join(line for line in value.splitlines() if not line.startswith(b"numpy_version = "))
    if path.exists():
        if normalize(path.read_bytes()) != normalize(content):
            raise ValueError(f"Existing generated metadata differs: {path}")
    else:
        path.write_bytes(content)


def plain(value):
    if dataclasses.is_dataclass(value):
        return {field.name: plain(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if hasattr(value, "_asdict"):
        return {key: plain(item) for key, item in value._asdict().items()}
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def toml_value(value) -> str:
    value = plain(value)
    if isinstance(value, dict):
        return "{" + ", ".join(json.dumps(key) + " = " + toml_value(item) for key, item in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    return json.dumps(value, allow_nan=False)
