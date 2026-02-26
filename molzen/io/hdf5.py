"""HDF5 molecule serialization utilities."""

from __future__ import annotations

import pickle
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


def _require_h5py():
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise ImportError("h5py is required for HDF5 support.") from exc
    return h5py


def _normalize_payload(data: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(data)
    extra = set(payload) - _ALLOWED_KEYS
    if extra:
        raise ValueError(f"Unsupported keys in molecule payload: {sorted(extra)}")
    payload.setdefault("metadata", {})
    return payload


def parse_hdf5(hdf5_fp: str) -> dict[str, Any]:
    """Load molecule data from an `.hdf5` file."""
    h5py = _require_h5py()
    with h5py.File(hdf5_fp, "r") as h5f:
        if "molzen_pickle" not in h5f:
            raise ValueError("HDF5 molecule file is missing dataset 'molzen_pickle'.")
        blob = bytes(np.asarray(h5f["molzen_pickle"], dtype=np.uint8).tolist())

    payload = pickle.loads(blob)
    if not isinstance(payload, dict):
        raise ValueError("HDF5 molecule payload must be a dictionary.")
    return _normalize_payload(payload)


def write_hdf5(filename: str, data: Mapping[str, Any]) -> None:
    """Write molecule data to `.hdf5`."""
    h5py = _require_h5py()
    payload = _normalize_payload(data)
    blob = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)

    with h5py.File(filename, "w") as h5f:
        h5f.attrs["format"] = "molzen.molecule"
        h5f.attrs["version"] = 1
        h5f.create_dataset("molzen_pickle", data=np.frombuffer(blob, dtype=np.uint8))
