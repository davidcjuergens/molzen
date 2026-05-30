"""Dependency-free DCD trajectory readers for coordinate payloads."""

from __future__ import annotations

import os
import struct
from typing import BinaryIO, Any

import numpy as np

from .rst7 import parse_prmtop, parse_qmindices

_DCD_MAGIC_VALUES = {b"CORD", b"VELD"}
_MAX_REASONABLE_RECORD_SIZE = 1024 * 1024 * 1024


def parse_dcd(dcd_fp: str) -> dict[str, Any]:
    """Parse coordinates from a DCD trajectory file.

    Args:
        dcd_fp: Path to the DCD trajectory file.
    """
    with open(dcd_fp, "rb") as f:
        endian = _detect_endian(f)
        header = _read_fortran_record(f, endian)
        if len(header) < 84 or header[:4] not in _DCD_MAGIC_VALUES:
            raise ValueError(f"Unsupported or malformed DCD header: {dcd_fp}")

        header_int_count = (len(header) - 4) // 4
        header_ints = struct.unpack(
            f"{endian}{header_int_count}i",
            header[4 : 4 + header_int_count * 4],
        )
        expected_n_frames = int(header_ints[0]) if header_ints else 0
        fixed_atom_count = int(header_ints[8]) if len(header_ints) > 8 else 0
        if fixed_atom_count:
            raise ValueError("DCD files with fixed atoms are not supported.")

        _ = _read_fortran_record(f, endian)  # title block
        natom_record = _read_fortran_record(f, endian)
        if len(natom_record) != 4:
            raise ValueError(f"Malformed DCD atom-count record: {dcd_fp}")
        n_atoms = struct.unpack(f"{endian}i", natom_record)[0]

        frames = _read_dcd_frames(f, endian, n_atoms)

    if not frames:
        raise ValueError(f"No coordinate frames found in DCD file: {dcd_fp}")

    return {
        "xyz": np.asarray(frames, dtype=float),
        "n_atoms": n_atoms,
        "n_frames": len(frames),
        "expected_n_frames": expected_n_frames,
    }


def parse_dcd_with_prmtop(
    dcd_fp: str,
    prmtop_fp: str,
    qmindices_fp: str | None = None,
) -> dict[str, Any]:
    """Parse a DCD trajectory and topology, optionally selecting QM atoms.

    Args:
        dcd_fp: Path to the DCD trajectory file.
        prmtop_fp: Path to the Amber topology file.
        qmindices_fp: Optional path to a TeraChem QM atom index file. If provided,
            only those atoms are included in the returned payload.
    """
    dcd_payload = parse_dcd(dcd_fp)
    top_payload = parse_prmtop(prmtop_fp)

    xyz = np.asarray(dcd_payload["xyz"], dtype=float)
    if xyz.shape[1] != int(top_payload["n_atoms"]):
        raise ValueError(
            f"Atom count mismatch: DCD has {xyz.shape[1]} atoms but prmtop has "
            f"{top_payload['n_atoms']} atoms."
        )

    if qmindices_fp is None:
        atom_indices = np.arange(xyz.shape[1], dtype=int)
    else:
        atom_indices = parse_qmindices(qmindices_fp)
        if np.any(atom_indices >= xyz.shape[1]):
            raise ValueError(
                f"qmindices references atom index {int(atom_indices.max())}, "
                f"but DCD only has {xyz.shape[1]} atoms."
            )

    selected = atom_indices.astype(int)
    metadata = {
        "dcd_path": os.path.abspath(dcd_fp),
        "prmtop_path": os.path.abspath(prmtop_fp),
        "source_atom_indices": selected.tolist(),
        "source_atom_index_base": 0,
        "dcd_n_frames": int(dcd_payload["n_frames"]),
        "dcd_expected_n_frames": int(dcd_payload["expected_n_frames"]),
    }
    if qmindices_fp is not None:
        metadata["qmindices_path"] = os.path.abspath(qmindices_fp)

    return {
        "xyz": xyz[:, selected, :],
        "atom_names": _select(top_payload["atom_names"], selected),
        "atom_types": _select(top_payload["atom_types"], selected),
        "elements": _select(top_payload["elements"], selected),
        "res_names": _select(top_payload["res_names"], selected),
        "res_nums": _select(top_payload["res_nums"], selected),
        "serials": (selected + 1).astype(int).tolist(),
        "metadata": metadata,
    }


def _detect_endian(f: BinaryIO) -> str:
    start = f.read(8)
    f.seek(0)
    if len(start) < 8:
        raise ValueError("DCD file is too short.")

    for endian in ("<", ">"):
        record_size = struct.unpack(f"{endian}i", start[:4])[0]
        if (
            0 < record_size < _MAX_REASONABLE_RECORD_SIZE
            and start[4:8] in _DCD_MAGIC_VALUES
        ):
            return endian

    raise ValueError("Could not determine DCD byte order.")


def _read_fortran_record(f: BinaryIO, endian: str) -> bytes:
    marker = f.read(4)
    if len(marker) != 4:
        raise EOFError("Unexpected end of DCD file while reading record marker.")

    record_size = struct.unpack(f"{endian}i", marker)[0]
    if record_size < 0 or record_size > _MAX_REASONABLE_RECORD_SIZE:
        raise ValueError(f"Invalid DCD record size: {record_size}")

    data = f.read(record_size)
    if len(data) != record_size:
        raise EOFError("Unexpected end of DCD file while reading record data.")

    end_marker = f.read(4)
    if len(end_marker) != 4:
        raise EOFError("Unexpected end of DCD file while reading record footer.")
    end_record_size = struct.unpack(f"{endian}i", end_marker)[0]
    if end_record_size != record_size:
        raise ValueError(
            f"DCD Fortran record marker mismatch: {record_size} != {end_record_size}"
        )
    return data


def _read_optional_fortran_record(f: BinaryIO, endian: str) -> bytes | None:
    marker = f.read(4)
    if marker == b"":
        return None
    if len(marker) != 4:
        raise EOFError("Unexpected end of DCD file while reading record marker.")
    f.seek(-4, os.SEEK_CUR)
    return _read_fortran_record(f, endian)


def _read_dcd_frames(f: BinaryIO, endian: str, n_atoms: int) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    float32_coord_bytes = n_atoms * 4
    float64_coord_bytes = n_atoms * 8

    while True:
        x_record = _read_optional_fortran_record(f, endian)
        if x_record is None:
            break

        # CHARMM/Amber DCD files may include a unit-cell record before each frame.
        if len(x_record) not in {float32_coord_bytes, float64_coord_bytes}:
            x_record = _read_fortran_record(f, endian)

        y_record = _read_fortran_record(f, endian)
        z_record = _read_fortran_record(f, endian)
        frames.append(
            np.column_stack(
                (
                    _decode_coord_record(x_record, endian, n_atoms),
                    _decode_coord_record(y_record, endian, n_atoms),
                    _decode_coord_record(z_record, endian, n_atoms),
                )
            )
        )

    return frames


def _decode_coord_record(record: bytes, endian: str, n_atoms: int) -> np.ndarray:
    if len(record) == n_atoms * 4:
        return np.frombuffer(record, dtype=f"{endian}f4").astype(float, copy=True)
    if len(record) == n_atoms * 8:
        return np.frombuffer(record, dtype=f"{endian}f8").astype(float, copy=True)
    raise ValueError(
        f"Unexpected DCD coordinate record size {len(record)} for {n_atoms} atoms."
    )


def _select(values: list[Any], indices: np.ndarray) -> list[Any]:
    return [values[int(i)] for i in indices]
