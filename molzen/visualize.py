"""Helpers for visualizing Molzen objects."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from molzen.io.molecule import Molecule


def _require_nglview() -> Any:
    """Import nglview lazily for optional visualization support."""
    try:
        import nglview as nv
    except ImportError as exc:
        raise ImportError(
            "nglview is required for Molecule.show(). "
            "Install molzen[viz] or pip install nglview. "
            f"Original import error: {exc}"
        ) from exc
    return nv


def _frame_count(atom_records: np.ndarray) -> int:
    """Return the number of coordinate frames in atom_records."""
    return atom_records.dtype["coords"].shape[0]


def _coerce_frame_index(atom_records: np.ndarray, frame: int | None) -> int | None:
    """Validate a requested frame index against atom_records."""
    if frame is None:
        return None

    n_frames = _frame_count(atom_records)
    if frame < 0 or frame >= n_frames:
        raise IndexError(f"Frame index {frame} out of range for {n_frames} frame(s).")
    return frame


def _viewer_atom_name(row: np.void, fallback_index: int) -> str:
    """Return a PDB-safe atom name for the viewer export."""
    atom_name = str(row["atom_name"]).strip()
    if atom_name:
        return atom_name[:4]

    element = str(row["element"]).strip()
    if element:
        return element[:4]
    return f"A{fallback_index}"[:4]


def _viewer_record_name(row: np.void) -> str:
    """Return the viewer record name, defaulting from entity kind."""
    record_name = str(row["record_name"]).strip().upper()
    if record_name:
        return record_name

    entity_kind = str(row["entity_kind"]).strip().lower()
    if entity_kind == "polymer":
        return "ATOM"
    return "HETATM"


def _pdb_lines_for_frame(atom_records: np.ndarray, frame_index: int) -> list[str]:
    """Serialize one coordinate frame to PDB atom lines."""
    lines: list[str] = []
    for i, row in enumerate(atom_records, start=1):
        record_name = _viewer_record_name(row)
        serial = int(row["serial"]) if int(row["serial"]) > 0 else int(row["atom_index"]) + 1
        atom_name = _viewer_atom_name(row, i)
        alt_loc = str(row["alt_loc"]).strip()[:1]
        res_name = (str(row["res_name"]).strip() or "MOL")[:3]
        chain_id = (str(row["chain_id"]).strip() or " ")[:1]
        res_num = int(row["res_num"]) if int(row["res_num"]) != 0 else 1
        i_code = str(row["i_code"]).strip()[:1]
        coord = np.asarray(row["coords"][frame_index], dtype=float)
        occupancy = float(row["occupancy"])
        if not np.isfinite(occupancy):
            occupancy = 1.00
        temp_factor = float(row["temp_factor"])
        if not np.isfinite(temp_factor):
            temp_factor = 0.00
        element = str(row["element"]).strip()[:2]
        charge = str(row["charge"]).strip()[:2]

        lines.append(
            f"{record_name:<6}{serial:5d} {atom_name:>4}{alt_loc:1}{res_name:>3} "
            f"{chain_id:1}{res_num:4d}{i_code:1}   "
            f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
            f"{occupancy:6.2f}{temp_factor:6.2f}          "
            f"{element:>2}{charge:>2}\n"
        )
    return lines


def _pdb_text(atom_records: np.ndarray, frame: int | None = None) -> str:
    """Serialize atom_records to PDB text for nglview."""
    if len(atom_records) == 0:
        raise ValueError("Cannot visualize an empty molecule.")

    frame = _coerce_frame_index(atom_records, frame)
    n_frames = _frame_count(atom_records)
    chunks: list[str] = []

    if frame is None and n_frames > 1:
        for frame_index in range(n_frames):
            chunks.append(f"MODEL     {frame_index + 1}\n")
            chunks.extend(_pdb_lines_for_frame(atom_records, frame_index))
            chunks.append("ENDMDL\n")
    else:
        frame_index = 0 if frame is None else frame
        chunks.extend(_pdb_lines_for_frame(atom_records, frame_index))

    chunks.append("END\n")
    return "".join(chunks)


def show_molecule(
    mol: Molecule,
    *,
    width: str = "300px",
    height: str = "300px",
    frame: int | None = None,
) -> Any:
    """Return an nglview widget for a molecule."""
    atom_records = mol.atom_records
    if atom_records is None or len(atom_records) == 0:
        raise ValueError("Cannot visualize an empty molecule.")

    nv = _require_nglview()
    pdb_text = _pdb_text(atom_records, frame=frame)
    view = nv.show_text(pdb_text)
    view.add_ball_and_stick()
    view.layout.width = width
    view.layout.height = height
    return view
