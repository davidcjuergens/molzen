"""Amber restart/prmtop readers for coordinate payloads."""

from __future__ import annotations

import os
import re
from typing import Any

import numpy as np

from molzen.ptable import ALL_SYMBOLS, z_to_symbol

_ELEMENT_SYMBOLS = set(ALL_SYMBOLS)
_FLOAT_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?")
_FORMAT_RE = re.compile(r"\((\d+)([A-Za-z])(\d+)(?:\.\d+)?\)")


def parse_rst7(rst7_fp: str) -> dict[str, Any]:
    """Parse coordinates from an Amber ASCII restart (`.rst7`) file.

    Args:
        rst7_fp: Path to the Amber restart file.
    """
    with open(rst7_fp, "r") as f:
        lines = f.readlines()

    if len(lines) < 2:
        raise ValueError(f"Amber restart is too short: {rst7_fp}")

    title = lines[0].rstrip("\n")
    header = lines[1].strip().split()
    if not header:
        raise ValueError(f"Missing atom count in Amber restart: {rst7_fp}")

    try:
        n_atoms = int(header[0])
    except ValueError as exc:
        raise ValueError(f"Invalid atom count in Amber restart: {header[0]!r}") from exc

    values: list[float] = []
    for line in lines[2:]:
        values.extend(
            float(match.group(0).replace("D", "E"))
            for match in _FLOAT_RE.finditer(line)
        )

    n_coord_values = n_atoms * 3
    if len(values) < n_coord_values:
        raise ValueError(
            f"Amber restart has {len(values)} coordinate values, expected "
            f"at least {n_coord_values} for {n_atoms} atoms."
        )

    xyz = np.asarray(values[:n_coord_values], dtype=float).reshape(n_atoms, 3)
    out: dict[str, Any] = {"xyz": xyz, "title": title, "n_atoms": n_atoms}
    if len(header) > 1:
        try:
            out["time"] = float(header[1].replace("D", "E"))
        except ValueError:
            pass
    extra_values = len(values) - n_coord_values
    if extra_values in {3, n_coord_values + 3}:
        out["box"] = np.asarray(values[-3:], dtype=float)
    return out


def parse_qmindices(qmindices_fp: str) -> np.ndarray:
    """Parse a TeraChem qmindices file into zero-based atom indices.

    Args:
        qmindices_fp: Path to the TeraChem QM atom index file.
    """
    with open(qmindices_fp, "r") as f:
        text = f.read()

    indices = [int(token) for token in re.findall(r"[+-]?\d+", text)]
    if not indices:
        raise ValueError(f"No atom indices found in qmindices file: {qmindices_fp}")

    arr = np.asarray(indices, dtype=int)
    if np.any(arr < 0):
        raise ValueError(f"qmindices must be non-negative/positive: {qmindices_fp}")

    return arr


def parse_prmtop(prmtop_fp: str) -> dict[str, Any]:
    """Parse enough Amber prmtop metadata to label restart coordinates.

    Args:
        prmtop_fp: Path to the Amber topology file.
    """
    sections = _parse_prmtop_sections(prmtop_fp)

    if "ATOM_NAME" not in sections:
        raise ValueError(f"Missing ATOM_NAME section in prmtop: {prmtop_fp}")

    atom_names = [str(name).strip() for name in sections["ATOM_NAME"]]
    n_atoms = len(atom_names)

    atom_types = [
        str(atom_type).strip()
        for atom_type in sections.get("AMBER_ATOM_TYPE", [""] * n_atoms)
    ]
    if len(atom_types) != n_atoms:
        atom_types = [""] * n_atoms

    atomic_numbers = sections.get("ATOMIC_NUMBER")
    elements = _elements_from_prmtop(atom_names, atom_types, atomic_numbers)

    res_names = ["MOL"] * n_atoms
    res_nums = [1] * n_atoms
    residue_labels = [str(label).strip() for label in sections.get("RESIDUE_LABEL", [])]
    residue_pointers = [int(pointer) for pointer in sections.get("RESIDUE_POINTER", [])]
    if residue_labels and residue_pointers:
        res_names, res_nums = _expand_residue_metadata(
            residue_labels,
            residue_pointers,
            n_atoms,
        )

    return {
        "atom_names": atom_names,
        "atom_types": atom_types,
        "elements": elements,
        "res_names": res_names,
        "res_nums": res_nums,
        "n_atoms": n_atoms,
    }


def parse_rst7_with_prmtop(
    rst7_fp: str,
    prmtop_fp: str,
    qmindices_fp: str | None = None,
) -> dict[str, Any]:
    """Parse an Amber restart and topology, optionally selecting QM atoms.

    Args:
        rst7_fp: Path to the Amber restart file.
        prmtop_fp: Path to the Amber topology file.
        qmindices_fp: Optional path to a TeraChem QM atom index file. If provided,
            only those atoms are included in the returned payload.
    """
    rst7_payload = parse_rst7(rst7_fp)
    top_payload = parse_prmtop(prmtop_fp)

    xyz = np.asarray(rst7_payload["xyz"], dtype=float)
    if xyz.shape[0] != int(top_payload["n_atoms"]):
        raise ValueError(
            f"Atom count mismatch: rst7 has {xyz.shape[0]} atoms but prmtop has "
            f"{top_payload['n_atoms']} atoms."
        )

    if qmindices_fp is None:
        atom_indices = np.arange(xyz.shape[0], dtype=int)
    else:
        atom_indices = parse_qmindices(qmindices_fp)
        if np.any(atom_indices >= xyz.shape[0]):
            raise ValueError(
                f"qmindices references atom index {int(atom_indices.max())}, "
                f"but rst7 only has {xyz.shape[0]} atoms."
            )

    selected = atom_indices.astype(int)
    metadata = {
        "rst7_path": os.path.abspath(rst7_fp),
        "prmtop_path": os.path.abspath(prmtop_fp),
        "source_atom_indices": selected.tolist(),
        "source_atom_index_base": 0,
        "rst7_title": rst7_payload.get("title", ""),
    }
    if qmindices_fp is not None:
        metadata["qmindices_path"] = os.path.abspath(qmindices_fp)
    if "box" in rst7_payload:
        metadata["rst7_box"] = rst7_payload["box"]
    if "time" in rst7_payload:
        metadata["rst7_time"] = rst7_payload["time"]

    return {
        "xyz": xyz[selected],
        "atom_names": _select(top_payload["atom_names"], selected),
        "atom_types": _select(top_payload["atom_types"], selected),
        "elements": _select(top_payload["elements"], selected),
        "res_names": _select(top_payload["res_names"], selected),
        "res_nums": _select(top_payload["res_nums"], selected),
        "serials": (selected + 1).astype(int).tolist(),
        "metadata": metadata,
    }


def _parse_prmtop_sections(prmtop_fp: str) -> dict[str, list[Any]]:
    sections: dict[str, list[Any]] = {}
    current_flag: str | None = None
    current_format: tuple[int, str, int] | None = None
    raw_data: list[str] = []

    def flush() -> None:
        nonlocal raw_data
        if current_flag is None or current_format is None:
            raw_data = []
            return
        sections[current_flag] = _parse_prmtop_values(raw_data, current_format)
        raw_data = []

    with open(prmtop_fp, "r") as f:
        for line in f:
            if line.startswith("%FLAG"):
                flush()
                parts = line.strip().split(maxsplit=1)
                current_flag = parts[1] if len(parts) > 1 else ""
                current_format = None
                continue
            if line.startswith("%FORMAT"):
                match = _FORMAT_RE.search(line)
                if match is None:
                    raise ValueError(f"Unsupported prmtop format line: {line.strip()}")
                current_format = (
                    int(match.group(1)),
                    match.group(2).upper(),
                    int(match.group(3)),
                )
                continue
            if current_flag is not None and current_format is not None:
                raw_data.append(line.rstrip("\n"))

    flush()
    return sections


def _parse_prmtop_values(lines: list[str], fmt: tuple[int, str, int]) -> list[Any]:
    _, kind, width = fmt
    values: list[Any] = []
    for line in lines:
        for start in range(0, len(line), width):
            field = line[start : start + width]
            if not field.strip():
                continue
            if kind == "A":
                values.append(field)
            elif kind == "I":
                values.append(int(field))
            elif kind in {"E", "F", "D"}:
                values.append(float(field.replace("D", "E")))
            else:
                raise ValueError(f"Unsupported prmtop format kind: {kind}")
    return values


def _expand_residue_metadata(
    residue_labels: list[str],
    residue_pointers: list[int],
    n_atoms: int,
) -> tuple[list[str], list[int]]:
    starts = [pointer - 1 for pointer in residue_pointers]
    if any(start < 0 or start >= n_atoms for start in starts):
        raise ValueError("Invalid RESIDUE_POINTER values in prmtop.")

    res_names = ["MOL"] * n_atoms
    res_nums = [1] * n_atoms
    for i, start in enumerate(starts):
        stop = starts[i + 1] if i + 1 < len(starts) else n_atoms
        label = residue_labels[i] if i < len(residue_labels) else "MOL"
        for atom_idx in range(start, stop):
            res_names[atom_idx] = label
            res_nums[atom_idx] = i + 1
    return res_names, res_nums


def _elements_from_prmtop(
    atom_names: list[str],
    atom_types: list[str],
    atomic_numbers: list[Any] | None,
) -> list[str]:
    if atomic_numbers is not None and len(atomic_numbers) == len(atom_names):
        out = []
        for z_value, atom_name, atom_type in zip(
            atomic_numbers,
            atom_names,
            atom_types,
            strict=False,
        ):
            z = int(z_value)
            out.append(z_to_symbol.get(z, _infer_element(atom_name, atom_type)))
        return out

    return [
        _infer_element(atom_name, atom_type)
        for atom_name, atom_type in zip(atom_names, atom_types, strict=False)
    ]


def _infer_element(atom_name: str, atom_type: str = "") -> str:
    for raw in (atom_type, atom_name):
        letters = "".join(ch for ch in raw.strip() if ch.isalpha())
        if not letters:
            continue

        first = letters[0].upper()
        if first in {"C", "H", "N", "O", "P", "S", "B", "F", "I"}:
            return first

        if len(letters) >= 2:
            cand2 = letters[:2].capitalize()
            if cand2 in _ELEMENT_SYMBOLS:
                return cand2

        cand1 = letters[0].upper()
        if cand1 in _ELEMENT_SYMBOLS:
            return cand1

    return ""


def _select(values: list[Any], indices: np.ndarray) -> list[Any]:
    return [values[int(i)] for i in indices]
