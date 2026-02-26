"""NPY molecule serialization utilities."""

from __future__ import annotations

import io
from typing import Any, Mapping

import numpy as np

_ALLOWED_KEYS = {
    "xyz",
    "atom_names",
    "elements",
    "comments",
    "seq",
    "hetatm",
    "metadata",
}


def _normalize_payload(data: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(data)
    extra = set(payload) - _ALLOWED_KEYS
    if extra:
        raise ValueError(f"Unsupported keys in molecule payload: {sorted(extra)}")
    payload.setdefault("metadata", {})
    return payload


def parse_npy(npy_fp: str) -> dict[str, Any]:
    """Load molecule data from a `.npy` file."""
    arr = np.load(npy_fp, allow_pickle=True)
    if arr.shape != () or arr.dtype != object:
        raise ValueError("Expected a pickled dictionary object in the NPY file.")

    payload = arr.item()
    if not isinstance(payload, dict):
        raise ValueError("NPY molecule payload must be a dictionary.")
    return _normalize_payload(payload)


def write_npy(
    filename: str, data: Mapping[str, Any], return_bytes: bool = False
) -> bytes | None:
    """Write molecule data to `.npy` or return NPY bytes."""
    payload = _normalize_payload(data)

    if return_bytes:
        buffer = io.BytesIO()
        np.save(buffer, payload, allow_pickle=True)
        return buffer.getvalue()

    np.save(filename, payload, allow_pickle=True)
    return None
