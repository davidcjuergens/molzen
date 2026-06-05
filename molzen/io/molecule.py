"""Lightweight standardized container for molecule data."""

from __future__ import annotations

from typing import Any, Iterator, Mapping

import numpy as np
import os

from molzen.amino_acids import aa2long, aa2num, aa_1_to_3, ncaas, oneletter_code
from molzen.ptable import symbol_to_z
from molzen.io.terachem.parse import parse_terachem_output
from . import dcd as dcd_io
from . import hdf5 as hdf5_io
from . import mol2 as mol2_io
from . import npy as npy_io
from . import pdb as pdb_io
from . import rst7 as rst7_io
from . import xyz as xyz_io

_MISSING = object()
_SOLVENT_RESNAMES = {"HOH", "WAT", "SOL", "H2O"}
_ION_TOKENS = {
    "AG",
    "AL",
    "BA",
    "BR",
    "CA",
    "CD",
    "CL",
    "CO",
    "CS",
    "CU",
    "FE",
    "HG",
    "IOD",
    "I",
    "K",
    "LI",
    "MG",
    "MN",
    "NA",
    "NI",
    "PB",
    "PT",
    "RB",
    "SR",
    "ZN",
}
_PDB_ATOM_INDEX = [
    {name.strip(): i for i, name in enumerate(long) if name is not None}
    for long in aa2long
]


def atom_record_dtype(n_frames: int) -> np.dtype:
    """Structured dtype for canonical per-atom records."""
    return np.dtype(
        [
            ("atom_index", "i4"),
            ("record_name", "U6"),
            ("entity_kind", "U8"),
            ("atom_name", "U8"),
            ("element", "U3"),
            ("res_name", "U4"),
            ("chain_id", "U4"),
            ("res_num", "i4"),
            ("i_code", "U1"),
            ("residue_index", "i4"),
            ("polymer_index", "i4"),
            ("alt_loc", "U1"),
            ("serial", "i4"),
            ("occupancy", "f4"),
            ("temp_factor", "f4"),
            ("charge", "U2"),
            ("atom_type", "U8"),
            ("coords", "f4", (n_frames, 3)),
        ]
    )


class Molecule(Mapping[str, Any]):
    """Canonical molecule container backed by atom_records."""

    _MAPPING_FIELDS = (
        "atom_records",
        "xyz",
        "polymer_xyz",
        "atom_names",
        "elements",
        "Z",
        "comments",
        "spinmult",
        "seq",
        "hetatm",
        "excited_state_records",
    )

    def __init__(
        self,
        xyz: np.ndarray | None = None,
        atom_names: list[str] | None = None,
        elements: list[str] | None = None,
        comments: list[str] | None = None,
        spinmult: int | str | None = None,
        seq: str | None = None,
        hetatm: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
        atom_records: np.ndarray | None = None,
        excited_state_records: list[dict[str, Any]] | None = None,
        _legacy_view: str | None = None,
    ) -> None:

        self._legacy_view = _legacy_view or self._infer_legacy_view(
            xyz=xyz,
            atom_names=atom_names,
            elements=elements,
            comments=comments,
            seq=seq,
            hetatm=hetatm,
            metadata=metadata,
        )

        self._atom_records: np.ndarray | None = None
        self._comments: list[str] | None = None
        self._spinmult: int | None = None
        self._metadata: dict[str, Any] = {}
        self._excited_state_records: list[dict[str, Any]] | None = None
        self.excited_state_records = excited_state_records

        self.comments = comments
        self.spinmult = spinmult
        if atom_records is not None:
            self.atom_records = atom_records
        else:
            records = self._build_atom_records_from_legacy(
                xyz=xyz,
                atom_names=atom_names,
                elements=elements,
                seq=seq,
                hetatm=hetatm,
                metadata=metadata,
                legacy_view=self._legacy_view,
            )
            self._set_atom_records(records)
        self.metadata = {} if metadata is None else metadata

    def _infer_legacy_view(
        self,
        *,
        xyz: np.ndarray | None,
        atom_names: list[str] | None,
        elements: list[str] | None,
        comments: list[str] | None,
        seq: str | None,
        hetatm: np.ndarray | None,
        metadata: dict[str, Any] | None,
    ) -> str:
        """Infer which legacy shape conventions should be preserved."""
        if metadata and ("pdb_records" in metadata or "pdb_raw_lines" in metadata):
            return "pdb"
        if seq is not None:
            return "pdb"
        if hetatm is not None and getattr(hetatm.dtype, "names", None):
            dtype_names = set(hetatm.dtype.names or ())
            if {"atom_idx", "atom_type", "element"}.issubset(dtype_names):
                return "mol2"
            if {"chain_id", "res_num"}.issubset(dtype_names):
                return "pdb"
        if comments is not None:
            return "xyz"
        if atom_names is not None or elements is not None:
            return "mol2"
        if xyz is not None:
            xyz_arr = np.asarray(xyz, dtype=float)
            if xyz_arr.ndim == 3 and xyz_arr.shape[-1] == 3:
                return "xyz"
        return "xyz"

    def _set_atom_records(self, atom_records: np.ndarray) -> None:
        """Validate and store canonical atom records."""
        records = self._coerce_atom_records(atom_records)
        if self._comments is not None and len(self._comments) != self._frame_count(
            records
        ):
            raise ValueError("comments length must match number of coordinate frames.")
        self._validate_excited_state_frame_indices(self._frame_count(records))
        self._atom_records = records

    def _validate_excited_state_frame_indices(self, n_frames: int) -> None:
        """Validate frame-aligned excited-state records against coordinate frames."""
        if self._excited_state_records is None:
            return

        for i, record in enumerate(self._excited_state_records):
            if "frame_index" not in record:
                continue
            try:
                frame_index = float(record["frame_index"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "excited_state_records frame_index values must be numeric."
                ) from exc

            if not frame_index.is_integer():
                raise ValueError(
                    "excited_state_records frame_index values must be integers."
                )

            frame_index_int = int(frame_index)
            if frame_index_int < 0 or frame_index_int >= n_frames:
                raise ValueError(
                    "excited_state_records frame_index out of range: "
                    f"record {i} has frame_index={frame_index_int}, "
                    f"but molecule has {n_frames} frame(s)."
                )

    @property
    def shape(self):
        """Return the shape of the xyz coordinates as (n_frames, n_atoms, 3)."""
        return self.xyz.shape

    def pop(self, idx: int) -> np.void:
        """Remove and return one atom record by index."""
        if self._atom_records is None:
            raise ValueError("No atom_records available to pop.")

        removed = self._atom_records[idx].copy()
        self._atom_records = np.delete(self._atom_records, idx, axis=0)
        self._clear_stale_pdb_metadata()
        return removed

    @property
    def dmap(self) -> np.ndarray:
        """Return the distance map of the molecule."""
        if self._atom_records is None:
            raise ValueError("Atom records are not set.")

        # Canonical coords in self._atom_records are atom-major (n_atoms, n_frames, 3)
        xyz = np.swapaxes(self._atom_records["coords"], 0, 1)
        T, natom, _ = xyz.shape

        # subtract all i from all j
        diff = xyz[:, :, None, :] - xyz[:, None, :, :]
        dmap = np.linalg.norm(diff, axis=-1)
        assert dmap.shape == (T, natom, natom)

        return dmap

    @staticmethod
    def _frame_count(atom_records: np.ndarray) -> int:
        """Return the number of coordinate frames encoded in the dtype."""
        return atom_records.dtype["coords"].shape[0]

    def _coerce_atom_records(self, atom_records: np.ndarray) -> np.ndarray:
        """Cast an arbitrary structured array into the canonical dtype."""
        records = np.asarray(atom_records)
        dtype_names = set(records.dtype.names or ())
        required = set(atom_record_dtype(1).names or ())
        if not required.issubset(dtype_names):
            missing = sorted(required - dtype_names)
            raise ValueError(f"atom_records is missing required fields: {missing}")

        coords = np.asarray(records["coords"], dtype=float)
        if coords.ndim != 3 or coords.shape[-1] != 3:
            raise ValueError("atom_records['coords'] must have shape (N, n_frames, 3).")

        n_frames = coords.shape[1]
        canonical = np.zeros(records.shape, dtype=atom_record_dtype(n_frames))
        for name in canonical.dtype.names or ():
            canonical[name] = records[name]
        return canonical

    @staticmethod
    def _normalize_atom_major_xyz(xyz: np.ndarray) -> np.ndarray:
        """Normalize atom-major coordinates to shape (n_frames, n_atoms, 3)."""
        coords = np.asarray(xyz, dtype=float)
        if coords.ndim == 2 and coords.shape[1] == 3:
            return coords[None, ...]
        if coords.ndim == 3 and coords.shape[2] == 3:
            return coords
        raise ValueError("xyz must have shape (N, 3) or (B, N, 3).")

    @staticmethod
    def _normalize_comments(
        comments: list[str] | None, n_frames: int
    ) -> list[str] | None:
        if comments is None:
            return None
        if len(comments) != n_frames:
            raise ValueError("comments length must match number of coordinate frames.")
        return list(comments)

    @staticmethod
    def _normalize_spinmult(spinmult: int | str | None) -> int | None:
        """Normalize spin multiplicity to a positive integer."""
        if spinmult is None:
            return None
        if isinstance(spinmult, bool):
            raise ValueError("spinmult must be a positive integer.")
        if not isinstance(spinmult, (int, np.integer, str)):
            raise ValueError("spinmult must be a positive integer.")
        try:
            value = int(spinmult)
        except (TypeError, ValueError) as exc:
            raise ValueError("spinmult must be a positive integer.") from exc
        if value < 1:
            raise ValueError("spinmult must be a positive integer.")
        if isinstance(spinmult, str) and str(value) != spinmult.strip():
            raise ValueError("spinmult must be a positive integer.")
        return value

    @staticmethod
    def _infer_entity_kind(record_name: str, res_name: str, element: str = "") -> str:
        """Assign a coarse entity label from record-level metadata."""
        record_name = record_name.strip().upper()
        res_name = res_name.strip().upper()
        element = element.strip().upper()
        if record_name == "ATOM":
            return "polymer"
        if res_name in _SOLVENT_RESNAMES:
            return "solvent"
        if res_name in _ION_TOKENS or element in _ION_TOKENS:
            return "ion"
        if record_name == "HETATM":
            return "ligand"
        return "unknown"

    @staticmethod
    def _assign_residue_indices(atom_records: np.ndarray) -> None:
        """Populate residue_index and polymer_index in first-seen order."""
        residue_lookup: dict[tuple[str, int, str, str, str], tuple[int, int]] = {}
        residue_counter = 0
        polymer_counter = 0

        for row in atom_records:
            is_polymer = (
                str(row["record_name"]).strip() == "ATOM"
                or str(row["entity_kind"]).strip() == "polymer"
            )
            key = (
                str(row["record_name"]).strip(),
                int(row["res_num"]),
                str(row["chain_id"]).strip(),
                str(row["i_code"]).strip(),
                str(row["res_name"]).strip(),
            )
            if key not in residue_lookup:
                polymer_index = polymer_counter if is_polymer else -1
                residue_lookup[key] = (residue_counter, polymer_index)
                residue_counter += 1
                if is_polymer:
                    polymer_counter += 1
            residue_index, polymer_index = residue_lookup[key]
            row["residue_index"] = residue_index
            row["polymer_index"] = polymer_index if is_polymer else -1

    def _build_atom_records_from_atom_major(
        self,
        *,
        xyz: np.ndarray,
        atom_names: list[str] | None = None,
        elements: list[str] | None = None,
        record_name: str = "HETATM",
        entity_kind: str = "unknown",
        res_names: list[str] | None = None,
        chain_ids: list[str] | None = None,
        res_nums: list[int] | None = None,
        i_codes: list[str] | None = None,
        alt_locs: list[str] | None = None,
        serials: list[int] | None = None,
        occupancies: list[float] | None = None,
        temp_factors: list[float] | None = None,
        charges: list[str] | None = None,
        atom_types: list[str] | None = None,
    ) -> np.ndarray:
        """Build canonical atom_records from atom-major coordinate data."""
        coords = self._normalize_atom_major_xyz(xyz)
        n_frames, n_atoms, _ = coords.shape

        def _validate_length(values: list[Any] | None, field_name: str) -> None:
            if values is not None and len(values) != n_atoms:
                raise ValueError(f"{field_name} length must match xyz atom count.")

        for field_name, values in (
            ("atom_names", atom_names),
            ("elements", elements),
            ("res_names", res_names),
            ("chain_ids", chain_ids),
            ("res_nums", res_nums),
            ("i_codes", i_codes),
            ("alt_locs", alt_locs),
            ("serials", serials),
            ("occupancies", occupancies),
            ("temp_factors", temp_factors),
            ("charges", charges),
            ("atom_types", atom_types),
        ):
            _validate_length(values, field_name)

        atom_records = np.zeros(n_atoms, dtype=atom_record_dtype(n_frames))
        atom_records["atom_index"] = np.arange(n_atoms, dtype=int)
        atom_records["record_name"] = record_name
        atom_records["entity_kind"] = entity_kind
        atom_records["atom_name"] = (
            atom_names if atom_names is not None else [""] * n_atoms
        )
        atom_records["element"] = elements if elements is not None else [""] * n_atoms
        atom_records["res_name"] = (
            res_names if res_names is not None else ["MOL"] * n_atoms
        )
        atom_records["chain_id"] = (
            chain_ids if chain_ids is not None else [""] * n_atoms
        )
        atom_records["res_num"] = res_nums if res_nums is not None else [1] * n_atoms
        atom_records["i_code"] = i_codes if i_codes is not None else [""] * n_atoms
        atom_records["alt_loc"] = alt_locs if alt_locs is not None else [""] * n_atoms
        atom_records["serial"] = (
            serials if serials is not None else np.arange(1, n_atoms + 1, dtype=int)
        )
        atom_records["occupancy"] = (
            occupancies if occupancies is not None else [np.nan] * n_atoms
        )
        atom_records["temp_factor"] = (
            temp_factors if temp_factors is not None else [np.nan] * n_atoms
        )
        atom_records["charge"] = charges if charges is not None else [""] * n_atoms
        atom_records["atom_type"] = (
            atom_types if atom_types is not None else [""] * n_atoms
        )
        atom_records["coords"] = np.swapaxes(coords, 0, 1)
        self._assign_residue_indices(atom_records)
        return atom_records

    def _build_atom_records_from_xyz_payload(
        self,
        *,
        xyz: np.ndarray,
        elements: list[str] | None,
    ) -> np.ndarray:
        """Adapt XYZ-style payload data into canonical atom_records."""
        return self._build_atom_records_from_atom_major(
            xyz=xyz,
            elements=elements,
            record_name="HETATM",
            entity_kind="unknown",
        )

    def _build_atom_records_from_mol2_payload(
        self,
        *,
        xyz: np.ndarray | None,
        atom_names: list[str] | None,
        elements: list[str] | None,
        hetatm: np.ndarray | None,
    ) -> np.ndarray:
        """Adapt MOL2-style payload data into canonical atom_records."""
        if hetatm is not None and len(hetatm) > 0:
            dtype_names = set(hetatm.dtype.names or ())
            required = {
                "atom_idx",
                "atom_name",
                "atom_type",
                "element",
                "res_name",
                "xyz",
            }
            if required.issubset(dtype_names):
                coords = np.asarray(hetatm["xyz"], dtype=float)
                atom_records = self._build_atom_records_from_atom_major(
                    xyz=coords,
                    atom_names=hetatm["atom_name"].tolist(),
                    elements=hetatm["element"].tolist(),
                    record_name="HETATM",
                    entity_kind="ligand",
                    res_names=hetatm["res_name"].tolist(),
                    serials=hetatm["atom_idx"].astype(int).tolist(),
                    atom_types=hetatm["atom_type"].tolist(),
                )
                atom_records["res_num"] = 1
                atom_records["residue_index"] = 0
                atom_records["polymer_index"] = -1
                return atom_records

        if xyz is None:
            raise ValueError("xyz is required when MOL2 hetatm data is not provided.")
        return self._build_atom_records_from_atom_major(
            xyz=xyz,
            atom_names=atom_names,
            elements=elements,
            record_name="HETATM",
            entity_kind="ligand",
            atom_types=elements,
        )

    def _build_atom_records_from_pdb_records(
        self, pdb_records: np.ndarray
    ) -> np.ndarray:
        """Convert parsed PDB ATOM/HETATM records into canonical atom_records."""
        atom_records = np.zeros(len(pdb_records), dtype=atom_record_dtype(1))
        atom_records["atom_index"] = np.arange(len(pdb_records), dtype=int)
        atom_records["record_name"] = pdb_records["record_name"]
        atom_records["atom_name"] = pdb_records["atom_name"]
        atom_records["element"] = pdb_records["element"]
        atom_records["res_name"] = pdb_records["res_name"]
        atom_records["chain_id"] = pdb_records["chain_id"]
        atom_records["res_num"] = pdb_records["res_num"]
        atom_records["i_code"] = pdb_records["i_code"]
        atom_records["alt_loc"] = pdb_records["alt_loc"]
        atom_records["serial"] = pdb_records["serial"]
        atom_records["occupancy"] = pdb_records["occupancy"]
        atom_records["temp_factor"] = pdb_records["temp_factor"]
        atom_records["charge"] = pdb_records["charge"]
        atom_records["coords"][:, 0, :] = np.asarray(pdb_records["xyz"], dtype=float)
        atom_records["entity_kind"] = [
            self._infer_entity_kind(record_name, res_name, element)
            for record_name, res_name, element in zip(
                pdb_records["record_name"],
                pdb_records["res_name"],
                pdb_records["element"],
                strict=False,
            )
        ]
        self._assign_residue_indices(atom_records)
        return atom_records

    def _build_atom_records_from_pdb_payload(
        self,
        *,
        xyz: np.ndarray | None,
        seq: str | None,
        hetatm: np.ndarray | None,
        metadata: dict[str, Any] | None,
    ) -> np.ndarray:
        """Adapt PDB-style residue and HETATM payloads into atom_records."""
        if metadata and "pdb_records" in metadata:
            return self._build_atom_records_from_pdb_records(metadata["pdb_records"])

        record_count = 0
        if xyz is not None and seq is not None:
            grid = np.asarray(xyz, dtype=float)
            if grid.ndim != 3 or grid.shape[2] != 3:
                raise ValueError("PDB xyz must have shape (Nres, Natom, 3).")
            if len(seq) != grid.shape[0]:
                raise ValueError(
                    "Sequence length must match number of residues in xyz."
                )
            for i, seq_token in enumerate(seq):
                res_name = aa_1_to_3[seq_token] if len(seq_token) == 1 else seq_token
                atom_map = _PDB_ATOM_INDEX[aa2num[res_name]]
                for atom_name, atom_idx in atom_map.items():
                    if not np.any(np.isnan(grid[i, atom_idx])):
                        record_count += 1

        if hetatm is not None:
            record_count += len(hetatm)

        atom_records = np.zeros(record_count, dtype=atom_record_dtype(1))
        cursor = 0

        if xyz is not None and seq is not None:
            grid = np.asarray(xyz, dtype=float)
            for residue_index, seq_token in enumerate(seq):
                res_name = aa_1_to_3[seq_token] if len(seq_token) == 1 else seq_token
                atom_map = _PDB_ATOM_INDEX[aa2num[res_name]]
                for atom_name, atom_idx in atom_map.items():
                    coord = grid[residue_index, atom_idx]
                    if np.any(np.isnan(coord)):
                        continue
                    atom_records[cursor]["atom_index"] = cursor
                    atom_records[cursor]["record_name"] = "ATOM"
                    atom_records[cursor]["entity_kind"] = "polymer"
                    atom_records[cursor]["atom_name"] = atom_name
                    atom_records[cursor]["element"] = pdb_io._infer_element(
                        atom_name, atom_name
                    )
                    atom_records[cursor]["res_name"] = res_name
                    atom_records[cursor]["chain_id"] = ""
                    atom_records[cursor]["res_num"] = residue_index + 1
                    atom_records[cursor]["i_code"] = ""
                    atom_records[cursor]["alt_loc"] = ""
                    atom_records[cursor]["serial"] = cursor + 1
                    atom_records[cursor]["occupancy"] = np.nan
                    atom_records[cursor]["temp_factor"] = np.nan
                    atom_records[cursor]["charge"] = ""
                    atom_records[cursor]["atom_type"] = ""
                    atom_records[cursor]["coords"][0] = coord
                    cursor += 1

        if hetatm is not None:
            dtype_names = set(hetatm.dtype.names or ())
            required = {"atom_name", "res_name", "chain_id", "res_num", "xyz"}
            if not required.issubset(dtype_names):
                missing = sorted(required - dtype_names)
                raise ValueError(
                    f"PDB hetatm array is missing required fields: {missing}"
                )

            for row in hetatm:
                atom_name = str(row["atom_name"])
                res_name = str(row["res_name"])
                chain_id = str(row["chain_id"])
                res_num = int(row["res_num"])
                coord = np.asarray(row["xyz"], dtype=float)
                atom_records[cursor]["atom_index"] = cursor
                atom_records[cursor]["record_name"] = "HETATM"
                atom_records[cursor]["entity_kind"] = self._infer_entity_kind(
                    "HETATM",
                    res_name,
                    pdb_io._infer_element(atom_name, atom_name),
                )
                atom_records[cursor]["atom_name"] = atom_name
                atom_records[cursor]["element"] = pdb_io._infer_element(
                    atom_name, atom_name
                )
                atom_records[cursor]["res_name"] = res_name
                atom_records[cursor]["chain_id"] = chain_id
                atom_records[cursor]["res_num"] = res_num
                atom_records[cursor]["i_code"] = ""
                atom_records[cursor]["alt_loc"] = ""
                atom_records[cursor]["serial"] = cursor + 1
                atom_records[cursor]["occupancy"] = np.nan
                atom_records[cursor]["temp_factor"] = np.nan
                atom_records[cursor]["charge"] = ""
                atom_records[cursor]["atom_type"] = ""
                atom_records[cursor]["coords"][0] = coord
                cursor += 1

        self._assign_residue_indices(atom_records)
        return atom_records

    def _build_atom_records_from_legacy(
        self,
        *,
        xyz: np.ndarray | None,
        atom_names: list[str] | None,
        elements: list[str] | None,
        seq: str | None,
        hetatm: np.ndarray | None,
        metadata: dict[str, Any] | None,
        legacy_view: str,
    ) -> np.ndarray:
        """Dispatch legacy payload conversion by source-style view."""
        if legacy_view == "pdb":
            return self._build_atom_records_from_pdb_payload(
                xyz=xyz,
                seq=seq,
                hetatm=hetatm,
                metadata=metadata,
            )
        if legacy_view == "mol2":
            return self._build_atom_records_from_mol2_payload(
                xyz=xyz,
                atom_names=atom_names,
                elements=elements,
                hetatm=hetatm,
            )
        if xyz is None:
            raise ValueError("xyz is required when atom_records is not provided.")
        return self._build_atom_records_from_xyz_payload(xyz=xyz, elements=elements)

    def _replace_coords(self, xyz: np.ndarray) -> None:
        """Replace canonical coordinates while preserving atom metadata."""
        if self._atom_records is None:
            raise ValueError("atom_records must exist before setting xyz directly.")

        coords = self._normalize_atom_major_xyz(xyz)
        n_frames, n_atoms, _ = coords.shape
        if n_atoms != len(self._atom_records):
            raise ValueError("xyz atom count must match existing atom_records.")
        if self._comments is not None and len(self._comments) != n_frames:
            raise ValueError("comments length must match number of coordinate frames.")

        updated = np.zeros(len(self._atom_records), dtype=atom_record_dtype(n_frames))
        for name in updated.dtype.names or ():
            if name == "coords":
                continue
            updated[name] = self._atom_records[name]
        updated["coords"] = np.swapaxes(coords, 0, 1)
        self._atom_records = updated
        self._clear_stale_pdb_metadata()

    def _clear_stale_pdb_metadata(self) -> None:
        """Drop cached raw PDB metadata after canonical edits."""
        self._metadata.pop("pdb_raw_lines", None)
        self._metadata.pop("pdb_records", None)

    def _update_from_legacy(
        self,
        *,
        xyz: Any = _MISSING,
        atom_names: Any = _MISSING,
        elements: Any = _MISSING,
        seq: Any = _MISSING,
        hetatm: Any = _MISSING,
        legacy_view: str | None = None,
    ) -> None:
        """Rebuild atom_records from a legacy compatibility view."""
        target_view = legacy_view or self._legacy_view

        if target_view == "pdb":
            payload = self._legacy_pdb_view()
            new_xyz = payload["polymer_xyz"] if xyz is _MISSING else xyz
            new_seq = self.seq if seq is _MISSING else seq
            new_hetatm = payload["hetatm"] if hetatm is _MISSING else hetatm

            if new_seq in (None, "") and (new_xyz is _MISSING or new_xyz is None):
                new_xyz = np.empty((0, 27, 3), dtype=float)
            records = self._build_atom_records_from_pdb_payload(
                xyz=new_xyz,
                seq=new_seq,
                hetatm=new_hetatm,
                metadata=None,
            )
            self._legacy_view = "pdb"
        elif target_view == "mol2":
            new_xyz = self._atom_major_xyz() if xyz is _MISSING else xyz
            new_atom_names = self.atom_names if atom_names is _MISSING else atom_names
            new_elements = self.elements if elements is _MISSING else elements
            new_hetatm = self._legacy_mol2_hetatm() if hetatm is _MISSING else hetatm
            records = self._build_atom_records_from_mol2_payload(
                xyz=new_xyz,
                atom_names=new_atom_names,
                elements=new_elements,
                hetatm=new_hetatm,
            )
            self._legacy_view = "mol2"
        else:
            new_xyz = self._atom_major_xyz() if xyz is _MISSING else xyz
            new_elements = self.elements if elements is _MISSING else elements
            records = self._build_atom_records_from_xyz_payload(
                xyz=new_xyz,
                elements=new_elements,
            )
            if atom_names is not _MISSING and atom_names is not None:
                records["atom_name"] = atom_names
            self._legacy_view = "xyz"

        self._set_atom_records(records)
        self._clear_stale_pdb_metadata()

    def _atom_major_xyz(self) -> np.ndarray | None:
        """Return coordinates in atom-major XYZ/MOL2 layout."""
        if self._atom_records is None:
            return None
        coords = np.swapaxes(self._atom_records["coords"], 0, 1)
        if coords.shape[0] == 1:
            return coords[0]
        return coords

    def _polymer_mask(self) -> np.ndarray:
        if self._atom_records is None:
            return np.zeros(0, dtype=bool)
        return (self._atom_records["record_name"] == "ATOM") | (
            self._atom_records["entity_kind"] == "polymer"
        )

    def _polymer_residue_order(self) -> list[int]:
        if self._atom_records is None:
            return []
        seen: set[int] = set()
        order: list[int] = []
        for row in self._atom_records[self._polymer_mask()]:
            residue_index = int(row["polymer_index"])
            if residue_index < 0 or residue_index in seen:
                continue
            seen.add(residue_index)
            order.append(residue_index)
        return order

    def _polymer_residue_rows(self) -> list[np.ndarray]:
        polymer_rows = self._atom_records[self._polymer_mask()]
        return [
            polymer_rows[polymer_rows["polymer_index"] == residue_index]
            for residue_index in self._polymer_residue_order()
        ]

    @staticmethod
    def _seq_token_from_res_name(res_name: str) -> str | None:
        res_name = res_name.strip().upper()
        token = oneletter_code.get(res_name)
        if token is not None:
            return token
        ncaa = ncaas.get(res_name)
        if ncaa is not None:
            return ncaa.get("canonical_one_letter")
        return None

    def _legacy_pdb_view(self) -> dict[str, Any]:
        """Derive residue-grid polymer data and PDB HETATM rows."""
        n_frames = (
            self._frame_count(self._atom_records)
            if self._atom_records is not None
            else 1
        )
        if n_frames != 1:
            raise ValueError(
                "PDB compatibility views require single-frame coordinates."
            )

        polymer_rows = self._polymer_residue_rows()
        polymer_xyz = np.full((len(polymer_rows), 27, 3), np.nan, dtype=float)
        seq_tokens: list[str] = []
        chains: list[str] = []

        for i, residue_rows in enumerate(polymer_rows):
            res_name = str(residue_rows[0]["res_name"]).strip()
            seq_tokens.append(self._seq_token_from_res_name(res_name) or res_name)
            chains.append(str(residue_rows[0]["chain_id"]).strip())
            atom_map = _PDB_ATOM_INDEX[aa2num[res_name]]
            for row in residue_rows:
                atom_name = str(row["atom_name"]).strip()
                atom_idx = atom_map.get(atom_name)
                if atom_idx is None:
                    continue
                polymer_xyz[i, atom_idx] = np.round(
                    np.asarray(row["coords"][0], dtype=float),
                    decimals=3,
                )

        hetatm_rows = self._atom_records[self._atom_records["record_name"] == "HETATM"]
        if len(hetatm_rows):
            hetatm = np.zeros(len(hetatm_rows), dtype=pdb_io.HETATM_DTYPES)
            atom_names = []
            for i, row in enumerate(hetatm_rows, start=1):
                atom_name = str(row["atom_name"]).strip()
                if not atom_name:
                    atom_name = str(row["element"]).strip() or f"A{i}"
                atom_names.append(atom_name)
            hetatm["atom_name"] = atom_names
            hetatm["res_name"] = hetatm_rows["res_name"]
            hetatm["chain_id"] = hetatm_rows["chain_id"]
            hetatm["res_num"] = hetatm_rows["res_num"]
            hetatm["xyz"] = np.round(hetatm_rows["coords"][:, 0, :], decimals=3)
        else:
            hetatm = np.array([], dtype=pdb_io.HETATM_DTYPES)

        return {
            "polymer_xyz": polymer_xyz,
            "seq_tokens": seq_tokens,
            "chains": chains,
            "hetatm": hetatm,
        }

    def _legacy_mol2_hetatm(self) -> np.ndarray:
        """Derive MOL2-compatible atom rows from canonical atom_records."""
        if self._atom_records is None:
            return np.array([], dtype=mol2_io.MOL2_HETATM_DTYPES)
        if self._frame_count(self._atom_records) != 1:
            raise ValueError(
                "MOL2 compatibility views require single-frame coordinates."
            )

        rows = self._atom_records
        hetatm = np.zeros(len(rows), dtype=mol2_io.MOL2_HETATM_DTYPES)
        hetatm["atom_idx"] = np.where(
            rows["serial"] > 0, rows["serial"], rows["atom_index"] + 1
        )
        hetatm["atom_name"] = rows["atom_name"]
        hetatm["atom_type"] = np.where(
            rows["atom_type"] != "", rows["atom_type"], rows["element"]
        )
        hetatm["element"] = rows["element"]
        hetatm["res_name"] = np.where(rows["res_name"] != "", rows["res_name"], "MOL")
        hetatm["xyz"] = rows["coords"][:, 0, :]
        return hetatm

    def _legacy_serialization_payload(self) -> dict[str, Any]:
        """Build the old dict payload expected by NPY and HDF5 writers."""
        payload: dict[str, Any] = {}
        if self._legacy_view == "pdb":
            pdb_view = self._legacy_pdb_view()
            payload["xyz"] = pdb_view["polymer_xyz"]
            payload["seq"] = self.seq
            if len(pdb_view["hetatm"]):
                payload["hetatm"] = pdb_view["hetatm"]
        else:
            xyz = self._atom_major_xyz()
            if xyz is not None:
                payload["xyz"] = xyz
            if self.atom_names is not None:
                payload["atom_names"] = self.atom_names
            if self.elements is not None:
                payload["elements"] = self.elements
            if self.comments is not None:
                payload["comments"] = self.comments
            if self._legacy_view == "mol2":
                payload["hetatm"] = self._legacy_mol2_hetatm()

        if self.metadata:
            payload["metadata"] = self.metadata
        if self.spinmult is not None:
            payload["spinmult"] = self.spinmult
        if self.excited_state_records is not None:
            payload["excited_state_records"] = self.excited_state_records
        return payload

    def _present_items(self, include_atom_records: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for field in self._MAPPING_FIELDS:
            if field == "atom_records" and not include_atom_records:
                continue
            value = getattr(self, field)
            if value is not None:
                data[field] = value
        if self.metadata:
            data["metadata"] = self.metadata
        return data

    def slice_frames(
        self,
        start: int | slice | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> Molecule:
        """Return a new molecule with only the selected coordinate frames.

        Args:
            start: The starting index or slice for frame selection.
            stop: If start is an int, the ending index for frame selection (exclusive).
            step: If start is an int, the step size for frame selection.

        Raises:
            ValueError: If atom_records is not set or if the resulting slice is invalid.
            TypeError: If start is a slice and stop or step is provided.

        Returns:
            A new Molecule instance containing only the selected frames.
        """
        if self._atom_records is None:
            raise ValueError("No atom_records available to slice.")

        # resolve to a slice object
        if isinstance(start, slice):
            if stop is not None or step is not None:
                raise TypeError("stop and step must be omitted when start is a slice.")
            frames = start
        else:
            frames = slice(start, stop, step)

        n_total_frames = self._frame_count(self._atom_records)
        selected_frame_indices = np.arange(n_total_frames)[frames]

        # grab coordinates corresponding to desired slice
        sliced_coords = self._atom_records["coords"][:, frames, :]
        # create a new atom_records array to populate
        n_frames = sliced_coords.shape[1]
        atom_records = np.zeros(
            len(self._atom_records), dtype=atom_record_dtype(n_frames)
        )

        for name in atom_records.dtype.names or ():
            if name == "coords":
                continue
            atom_records[name] = self._atom_records[name]
        atom_records["coords"] = sliced_coords

        metadata = self._slice_metadata(selected_frame_indices)

        comments_out = None if self._comments is None else self._comments[frames]
        excited_state_records = self._slice_excited_state_records(
            selected_frame_indices
        )

        # make new Molecule instance with sliced data
        return Molecule(
            atom_records=atom_records,
            comments=comments_out,
            spinmult=self.spinmult,
            metadata=metadata,
            excited_state_records=excited_state_records,
            _legacy_view=self._legacy_view,
        )

    def _slice_metadata(self, selected_frame_indices: np.ndarray) -> dict[str, Any]:
        """Return metadata adjusted for a frame slice."""
        metadata = dict(self.metadata)
        metadata.pop("pdb_raw_lines", None)
        metadata.pop("pdb_records", None)

        cat_metadata = metadata.get("cat_frames")
        if isinstance(cat_metadata, dict):
            cat_metadata = dict(cat_metadata)
            raw_boundaries = cat_metadata.get("frame_boundaries")
            if raw_boundaries is not None:
                selected = np.asarray(selected_frame_indices, dtype=int)
                remapped_boundaries = []
                for raw_boundary in raw_boundaries:
                    try:
                        boundary = int(raw_boundary)
                    except (TypeError, ValueError):
                        continue
                    local_boundary = int(np.count_nonzero(selected < boundary))
                    if 0 < local_boundary < len(selected):
                        remapped_boundaries.append(local_boundary)
                cat_metadata["frame_boundaries"] = sorted(set(remapped_boundaries))
            metadata["cat_frames"] = cat_metadata

        return metadata

    @classmethod
    def cat_frames(cls, molecules: list[Molecule]) -> Molecule:
        """Concatenate molecules along the coordinate-frame axis.

        Args:
            molecules: Molecules with matching atom records to concatenate in order.

        Raises:
            ValueError: If no molecules are provided, any molecule lacks atom
                records, atom metadata does not match, or spin multiplicities conflict.
            TypeError: If any item is not a Molecule.

        Returns:
            A new Molecule containing all frames from the input molecules.
        """
        if not molecules:
            raise ValueError("At least one molecule is required.")
        if any(not isinstance(mol, Molecule) for mol in molecules):
            raise TypeError("All items must be Molecule instances.")

        first = molecules[0]
        if first.atom_records is None:
            raise ValueError("All molecules must have atom_records.")

        for i, mol in enumerate(molecules[1:], start=1):
            if mol.atom_records is None:
                raise ValueError("All molecules must have atom_records.")
            cls._validate_frame_concat_compatible(first, mol, index=i)

        n_frames = sum(cls._frame_count(mol.atom_records) for mol in molecules)
        atom_records = np.zeros(
            len(first.atom_records), dtype=atom_record_dtype(n_frames)
        )
        for name in atom_records.dtype.names or ():
            if name == "coords":
                continue
            atom_records[name] = first.atom_records[name]
        atom_records["coords"] = np.concatenate(
            [mol.atom_records["coords"] for mol in molecules],
            axis=1,
        )

        comments = cls._cat_frame_comments(molecules)
        excited_state_records = cls._cat_excited_state_records(molecules)
        spinmult = cls._cat_spinmult(molecules)
        metadata = cls._cat_frame_metadata(molecules)

        return cls(
            atom_records=atom_records,
            comments=comments,
            spinmult=spinmult,
            metadata=metadata,
            excited_state_records=excited_state_records,
            _legacy_view=first._legacy_view,
        )

    @classmethod
    def _validate_frame_concat_compatible(
        cls,
        first: Molecule,
        other: Molecule,
        *,
        index: int,
    ) -> None:
        """Validate that two molecules can be concatenated framewise."""
        first_records = first.atom_records
        other_records = other.atom_records
        if first_records is None or other_records is None:
            raise ValueError("All molecules must have atom_records.")
        if len(first_records) != len(other_records):
            raise ValueError(
                "Cannot concatenate molecules with different atom counts: "
                f"molecule 0 has {len(first_records)} atoms, molecule {index} has "
                f"{len(other_records)} atoms."
            )

        metadata_fields = [
            name for name in first_records.dtype.names or () if name != "coords"
        ]
        for field in metadata_fields:
            first_values = first_records[field]
            other_values = other_records[field]
            if np.issubdtype(first_values.dtype, np.floating):
                matches = np.allclose(first_values, other_values, equal_nan=True)
            else:
                matches = np.array_equal(first_values, other_values)
            if not matches:
                raise ValueError(
                    "Cannot concatenate molecules with different atom metadata: "
                    f"field {field!r} differs for molecule {index}."
                )

    @classmethod
    def _cat_frame_comments(cls, molecules: list[Molecule]) -> list[str] | None:
        """Concatenate comments, padding missing comments with empty strings."""
        if not any(mol.comments is not None for mol in molecules):
            return None

        comments: list[str] = []
        for mol in molecules:
            n_frames = cls._frame_count(mol.atom_records)
            if mol.comments is None:
                comments.extend([""] * n_frames)
            else:
                comments.extend(mol.comments)
        return comments

    @classmethod
    def _cat_excited_state_records(
        cls,
        molecules: list[Molecule],
    ) -> list[dict[str, Any]] | None:
        """Concatenate frame-aligned excited-state records with frame offsets."""
        records: list[dict[str, Any]] = []
        frame_offset = 0
        for mol in molecules:
            for record in mol.excited_state_records or []:
                if "frame_index" not in record:
                    continue
                new_record = dict(record)
                new_record["frame_index"] = (
                    int(new_record["frame_index"]) + frame_offset
                )
                records.append(new_record)
            frame_offset += cls._frame_count(mol.atom_records)
        return records or None

    @staticmethod
    def _cat_spinmult(molecules: list[Molecule]) -> int | None:
        """Return the common non-null spin multiplicity, if any."""
        spinmults = {mol.spinmult for mol in molecules if mol.spinmult is not None}
        if len(spinmults) > 1:
            raise ValueError("Cannot concatenate molecules with different spinmults.")
        return next(iter(spinmults)) if spinmults else None

    @classmethod
    def _cat_frame_metadata(cls, molecules: list[Molecule]) -> dict[str, Any]:
        """Build provenance metadata for a frame concatenation."""
        segments: list[dict[str, Any]] = []
        frame_boundaries: list[int] = []
        frame_start = 0
        for i, mol in enumerate(molecules):
            n_frames = cls._frame_count(mol.atom_records)
            if i > 0:
                frame_boundaries.append(frame_start)
            segments.append(
                {
                    "molecule_index": i,
                    "frame_start": frame_start,
                    "frame_stop": frame_start + n_frames,
                    "metadata": dict(mol.metadata),
                }
            )
            frame_start += n_frames
        return {
            "cat_frames": {"segments": segments, "frame_boundaries": frame_boundaries}
        }

    def _slice_excited_state_records(
        self, selected_frame_indices: np.ndarray
    ) -> list[dict[str, Any]] | None:
        """Return excited-state records for selected frames with remapped indices."""
        if self._excited_state_records is None:
            return None

        # Build a lookup from original frame numbers to the new local frame numbers
        # in the sliced molecule. For example, slicing frames 10:12 maps 10 -> 0
        # and 11 -> 1.
        old_to_new = {
            int(old_frame): int(new_frame)
            for new_frame, old_frame in enumerate(selected_frame_indices.tolist())
        }
        sliced_records: list[dict[str, Any]] = []
        for record in self._excited_state_records:
            # Records without a frame index cannot be safely associated with this
            # slice, so drop them instead of carrying stale frame-aligned data.
            if "frame_index" not in record:
                continue
            old_frame = int(record["frame_index"])
            if old_frame not in old_to_new:
                continue
            # Copy the record before mutating so the original molecule keeps its
            # original frame numbering.
            new_record = dict(record)
            new_record["frame_index"] = old_to_new[old_frame]
            sliced_records.append(new_record)
        return sliced_records

    def __getitem__(self, key: str | int | slice) -> Any:
        """Get a molecule property by key OR return a Molecule() with selected frames."""
        # Treat slice and integer syntax as frame selection before mapping-style access.
        if isinstance(key, slice):
            return self.slice_frames(key)

        # single frame selection with integer index
        if isinstance(key, (int, np.integer)) and not isinstance(key, bool):
            if self._atom_records is None:
                raise ValueError("No atom_records available to slice.")
            frame_index = int(key)
            n_frames = self._frame_count(self._atom_records)
            # get the positive frame index equivalent to the negative index
            if frame_index < 0:
                frame_index += n_frames
            if frame_index < 0 or frame_index >= n_frames:
                raise IndexError(
                    f"Frame index {key} out of range for {n_frames} frame(s)."
                )
            return self.slice_frames(frame_index, frame_index + 1)

        # Keep string-key lookups compatible with the Mapping interface.
        if key == "metadata":
            if self.metadata:
                return self.metadata
            raise KeyError(key)

        if key not in self._MAPPING_FIELDS:
            raise KeyError(key)

        # Hide absent optional fields the same way a dict would.
        value = getattr(self, key)
        if value is None:
            raise KeyError(key)
        return value

    def __iter__(self) -> Iterator[str]:
        return iter(self._present_items())

    def __len__(self) -> int:
        return len(self._present_items())

    def as_dict(
        self, include_none: bool = False, include_atom_records: bool = False
    ) -> dict[str, Any]:
        """Return a dictionary representation of the molecule."""
        if include_none:
            data = {
                field: (
                    self.atom_records
                    if field == "atom_records"
                    else getattr(self, field)
                )
                for field in self._MAPPING_FIELDS
                if include_atom_records or field != "atom_records"
            }
            data["metadata"] = self.metadata
            return data
        return self._present_items(include_atom_records=include_atom_records)

    def __repr__(self) -> str:
        parts: list[str] = []
        frame_count = 0

        if self.atom_records is not None:
            frame_count = self._frame_count(self.atom_records)
            parts.append(f"atom_records={len(self.atom_records)}")
            parts.append(f"frames={frame_count}")
        if self.xyz is not None:
            parts.append(f"xyz_shape={tuple(np.asarray(self.xyz).shape)}")
        if frame_count == 1 and self.polymer_xyz is not None:
            parts.append(f"polymer_xyz_shape={tuple(self.polymer_xyz.shape)}")
        if self.atom_names is not None:
            parts.append(f"atom_names={len(self.atom_names)}")
        if self.elements is not None:
            parts.append(f"elements={len(self.elements)}")
        if self.comments is not None:
            parts.append(f"comments={len(self.comments)}")
        if self.spinmult is not None:
            parts.append(f"spinmult={self.spinmult}")
        if self.seq is not None:
            parts.append(f"seq_len={len(self.seq)}")
        if self.atom_records is not None:
            hetatm_count = int(
                np.count_nonzero(self.atom_records["record_name"] == "HETATM")
            )
            if hetatm_count:
                parts.append(f"hetatm={hetatm_count}")
        if self.metadata:
            parts.append(f"metadata_keys={sorted(self.metadata)}")
        if self.excited_state_records is not None:
            parts.append(f"excited_state_records={len(self.excited_state_records)}")

        if not parts:
            return "Molecule()"
        return f"Molecule({', '.join(parts)})"

    @property
    def atom_records(self) -> np.ndarray | None:
        return self._atom_records

    @atom_records.setter
    def atom_records(self, value: np.ndarray | None) -> None:
        if value is None:
            self._atom_records = None
            return
        self._set_atom_records(value)

    @property
    def comments(self) -> list[str] | None:
        return None if self._comments is None else list(self._comments)

    @comments.setter
    def comments(self, value: list[str] | None) -> None:
        if value is None:
            self._comments = None
            return
        comments = list(value)
        if self._atom_records is not None and len(comments) != self._frame_count(
            self._atom_records
        ):
            raise ValueError("comments length must match number of coordinate frames.")
        self._comments = comments

    @property
    def spinmult(self) -> int | None:
        return self._spinmult

    @spinmult.setter
    def spinmult(self, value: int | str | None) -> None:
        self._spinmult = self._normalize_spinmult(value)

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @metadata.setter
    def metadata(self, value: dict[str, Any] | None) -> None:
        self._metadata = {} if value is None else dict(value)

    @property
    def xyz(self) -> np.ndarray | None:
        """Return atom-major coordinates for all atoms."""
        if self._atom_records is None:
            return None
        return self._atom_major_xyz()

    @xyz.setter
    def xyz(self, value: np.ndarray | None) -> None:
        if value is None:
            self._atom_records = None
            return
        if self._atom_records is None:
            self._legacy_view = "xyz"
            self._set_atom_records(
                self._build_atom_records_from_xyz_payload(
                    xyz=value,
                    elements=self.elements,
                )
            )
            return
        self._replace_coords(value)

    @property
    def polymer_xyz(self) -> np.ndarray | None:
        """Return residue-major polymer coordinates for PDB-style workflows."""
        if self._atom_records is None or not np.any(self._polymer_mask()):
            return None
        return self._legacy_pdb_view()["polymer_xyz"]

    @polymer_xyz.setter
    def polymer_xyz(self, value: np.ndarray | None) -> None:
        if value is None:
            return
        self._update_from_legacy(xyz=value, legacy_view="pdb")

    @property
    def atom_names(self) -> list[str] | None:
        if self._atom_records is None:
            return None
        names = [str(name) for name in self._atom_records["atom_name"].tolist()]
        if not any(name.strip() for name in names):
            return None
        return names

    @atom_names.setter
    def atom_names(self, value: list[str] | None) -> None:
        self._update_from_legacy(atom_names=value, legacy_view="mol2")

    @property
    def elements(self) -> list[str] | None:
        if self._atom_records is None:
            return None
        elements = [str(element) for element in self._atom_records["element"].tolist()]
        if not any(element.strip() for element in elements):
            return None
        return elements

    @elements.setter
    def elements(self, value: list[str] | None) -> None:
        target_view = "mol2" if self._legacy_view == "mol2" else "xyz"
        self._update_from_legacy(elements=value, legacy_view=target_view)

    @property
    def seq(self) -> str | None:
        if self._atom_records is None or not np.any(self._polymer_mask()):
            return None
        tokens = []
        for residue_rows in self._polymer_residue_rows():
            res_name = str(residue_rows[0]["res_name"]).strip()
            token = self._seq_token_from_res_name(res_name)
            if token is None:
                return None
            tokens.append(token)
        return "".join(tokens)

    @seq.setter
    def seq(self, value: str | None) -> None:
        if value in (None, "") and not np.any(self._polymer_mask()):
            self._legacy_view = "pdb"
            return
        self._update_from_legacy(seq=value, legacy_view="pdb")

    @property
    def hetatm(self) -> np.ndarray | None:
        if self._atom_records is None:
            return None
        hetatm_rows = self._atom_records[self._atom_records["record_name"] == "HETATM"]
        if not len(hetatm_rows):
            return None
        if self._legacy_view == "mol2":
            return self._legacy_mol2_hetatm()
        return self._legacy_pdb_view()["hetatm"]

    @hetatm.setter
    def hetatm(self, value: np.ndarray | None) -> None:
        target_view = "mol2" if self._legacy_view == "mol2" else "pdb"
        self._update_from_legacy(hetatm=value, legacy_view=target_view)

    @property
    def Z(self) -> np.ndarray | None:
        elements = self.elements
        if elements is None:
            return None
        return np.array([symbol_to_z[e.capitalize()] for e in elements], dtype=int)

    @property
    def excited_state_records(self) -> list[dict[str, Any]] | None:
        return (
            None
            if self._excited_state_records is None
            else [dict(record) for record in self._excited_state_records]
        )

    @excited_state_records.setter
    def excited_state_records(self, value: list[dict[str, Any]] | None) -> None:
        if value is None:
            self._excited_state_records = None
            return
        self._excited_state_records = [dict(record) for record in value]
        if self._atom_records is not None:
            self._validate_excited_state_frame_indices(
                self._frame_count(self._atom_records)
            )

    @staticmethod
    def _terachem_records_with_frame_indices(
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Map parser section indices onto molecule frame indices."""
        out: list[dict[str, Any]] = []
        for record in records:
            new_record = dict(record)
            # Current TeraChem optimization outputs appear to write excited-state
            # sections in the same order as frames in optim.xyz.
            if "frame_index" not in new_record and "section_idx" in new_record:
                try:
                    section_idx = float(new_record["section_idx"])
                except (TypeError, ValueError):
                    section_idx = np.nan
                # Keep records without a usable section index, but only assign
                # frame_index when the parser gave us a finite numeric section.
                if np.isfinite(section_idx):
                    new_record["frame_index"] = int(section_idx)
            out.append(new_record)
        return out

    def show(
        self,
        *,
        width: int | str = 500,
        height: int | str = 300,
        frame: int | None = None,
        start: int | None = None,
        end: int | None = None,
        export_controls: bool = False,
        gif_delay_ms: int = 120,
        gif_total_time: float | None = None,
        gif_bounce: bool = False,
    ) -> Any:
        """Return a py3Dmol view for the molecule.

        Args:
            width: Viewer width in pixels or a CSS size string.
            height: Viewer height in pixels or a CSS size string.
            frame: Optional frame index to show from the displayed frame range.
            start: Optional first frame index to include.
            end: Optional frame index at which to stop, exclusive.
            export_controls: Whether to add browser-side GIF export controls for
                multi-frame views.
            gif_delay_ms: Delay between exported GIF frames in milliseconds.
            gif_total_time: Total exported GIF duration in seconds. When provided,
                this overrides gif_delay_ms.
            gif_bounce: Whether exported GIF frames should play forward and then
                backward to the first frame.
        """

        # optional dependency, so lazy import
        from molzen.visualize import show_molecule

        mol = self if start is None and end is None else self.slice_frames(start, end)
        width_str = f"{width}px" if isinstance(width, int) else width
        height_str = f"{height}px" if isinstance(height, int) else height
        return show_molecule(
            mol,
            width=width_str,
            height=height_str,
            frame=frame,
            export_controls=export_controls,
            gif_delay_ms=gif_delay_ms,
            gif_total_time=gif_total_time,
            gif_bounce=gif_bounce,
        )

    @classmethod
    def from_xyz(cls, file_path: str) -> Molecule:
        """Load a molecule from an XYZ file path."""
        payload = xyz_io.parse_xyz(file_path)
        return cls(_legacy_view="xyz", **payload)

    def to_xyz(self, file_path: str, return_str: bool = False) -> str | None:
        """Write this molecule to XYZ format."""
        xyz = self._atom_major_xyz()
        elements = self.elements
        if xyz is None:
            raise ValueError("xyz is required for XYZ output.")
        if elements is None:
            raise ValueError("elements are required for XYZ output.")
        return xyz_io.write_xyz(
            file_path,
            xyz,
            elements,
            comments=self.comments,
            return_str=return_str,
        )

    @classmethod
    def from_pdb(cls, file_path: str) -> Molecule:
        """Load a molecule from a PDB file path."""
        payload = pdb_io.parse_pdb(file_path)
        return cls(_legacy_view="pdb", **payload)

    def to_pdb(
        self,
        file_path: str,
        chains: list[str] | None = None,
        return_str: bool = False,
        use_raw_metadata: bool = True,
    ) -> str | None:
        """Write this molecule to PDB format."""
        raw_lines = (
            self.metadata.get("pdb_raw_lines")
            if use_raw_metadata and self.metadata
            else None
        )
        if raw_lines is not None:
            return pdb_io.write_pdb(
                file_path,
                xyz=None,
                seq=None,
                chains=chains,
                hetatm=None,
                pdb_raw_lines=raw_lines,
                return_str=return_str,
            )

        pdb_view = self._legacy_pdb_view()
        pdb_xyz = pdb_view["polymer_xyz"]
        seq_tokens = pdb_view["seq_tokens"]
        pdb_chains = chains if chains is not None else pdb_view["chains"]
        return pdb_io.write_pdb(
            file_path,
            xyz=pdb_xyz,
            seq=seq_tokens,
            chains=pdb_chains,
            hetatm=pdb_view["hetatm"] if len(pdb_view["hetatm"]) else None,
            return_str=return_str,
        )

    @classmethod
    def from_mol2(cls, file_path: str) -> Molecule:
        """Load a molecule from a MOL2 file path."""
        payload = mol2_io.parse_mol2(file_path)
        return cls(_legacy_view="mol2", **payload)

    def to_mol2(self, file_path: str, return_str: bool = False) -> str | None:
        """Write this molecule to MOL2 format."""
        return mol2_io.write_mol2(
            file_path,
            xyz=None,
            atom_names=None,
            elements=None,
            hetatm=self._legacy_mol2_hetatm(),
            return_str=return_str,
        )

    @classmethod
    def from_rst7(
        cls,
        rst7_path: str,
        prmtop_path: str,
        qmindices_path: str | None = None,
    ) -> Molecule:
        """Load Amber restart coordinates with labels from a prmtop file.

        Args:
            rst7_path: Path to the Amber restart file.
            prmtop_path: Path to the Amber topology file.
            qmindices_path: Optional path to a TeraChem QM atom index file. If
                provided, only those atoms are included.
        """
        payload = rst7_io.parse_rst7_with_prmtop(
            rst7_path,
            prmtop_path,
            qmindices_fp=qmindices_path,
        )

        metadata = payload["metadata"]
        bootstrap = cls(
            xyz=payload["xyz"],
            elements=payload["elements"],
            metadata=metadata,
            _legacy_view="xyz",
        )
        atom_records = bootstrap._build_atom_records_from_atom_major(
            xyz=payload["xyz"],
            atom_names=payload["atom_names"],
            elements=payload["elements"],
            record_name="HETATM",
            entity_kind="unknown",
            res_names=payload["res_names"],
            res_nums=payload["res_nums"],
            serials=payload["serials"],
            atom_types=payload["atom_types"],
        )
        atom_records["entity_kind"] = [
            cls._infer_entity_kind(record_name, res_name, element)
            for record_name, res_name, element in zip(
                atom_records["record_name"],
                atom_records["res_name"],
                atom_records["element"],
                strict=False,
            )
        ]
        return cls(
            atom_records=atom_records,
            metadata=metadata,
            _legacy_view="xyz",
        )

    @classmethod
    def from_dcd(
        cls,
        dcd_path: str,
        prmtop_path: str,
        qmindices_path: str | None = None,
    ) -> Molecule:
        """Load DCD trajectory coordinates with labels from a prmtop file.

        Args:
            dcd_path: Path to the DCD trajectory file.
            prmtop_path: Path to the Amber topology file.
            qmindices_path: Optional path to a TeraChem QM atom index file. If
                provided, only those atoms are included.
        """
        payload = dcd_io.parse_dcd_with_prmtop(
            dcd_path,
            prmtop_path,
            qmindices_fp=qmindices_path,
        )

        metadata = payload["metadata"]
        bootstrap = cls(
            xyz=payload["xyz"],
            elements=payload["elements"],
            metadata=metadata,
            _legacy_view="xyz",
        )
        atom_records = bootstrap._build_atom_records_from_atom_major(
            xyz=payload["xyz"],
            atom_names=payload["atom_names"],
            elements=payload["elements"],
            record_name="HETATM",
            entity_kind="unknown",
            res_names=payload["res_names"],
            res_nums=payload["res_nums"],
            serials=payload["serials"],
            atom_types=payload["atom_types"],
        )
        atom_records["entity_kind"] = [
            cls._infer_entity_kind(record_name, res_name, element)
            for record_name, res_name, element in zip(
                atom_records["record_name"],
                atom_records["res_name"],
                atom_records["element"],
                strict=False,
            )
        ]
        return cls(
            atom_records=atom_records,
            metadata=metadata,
            _legacy_view="xyz",
        )

    @classmethod
    def from_npy(cls, file_path: str) -> Molecule:
        """Load a molecule from an NPY file path."""
        payload = npy_io.parse_npy(file_path)
        return cls(**payload)

    def to_npy(self, file_path: str, return_bytes: bool = False) -> bytes | None:
        """Write this molecule to NPY format."""
        return npy_io.write_npy(
            file_path,
            self._legacy_serialization_payload(),
            return_bytes,
        )

    @classmethod
    def from_hdf5(cls, file_path: str) -> Molecule:
        """Load a molecule from an HDF5 file path."""
        payload = hdf5_io.parse_hdf5(file_path)
        return cls(**payload)

    def to_hdf5(self, file_path: str) -> None:
        """Write this molecule to HDF5 format."""
        hdf5_io.write_hdf5(file_path, self._legacy_serialization_payload())

    @classmethod
    def from_terachem_stdout(
        cls,
        file_path: str,
        raw_str_in: bool = False,
        *,
        dcd_path: str | None = None,
        rst7_path: str | None = None,
        prmtop_path: str | None = None,
        qmindices_path: str | None = None,
    ) -> Molecule:
        """Load a molecule from a TeraChem stdout file path.

        Args:
            file_path: Path to a terachem stdout, or raw terachem stdout string
            raw_str_in: If True, treat file_path as a raw terachem stdout string instead of a file path.
            dcd_path: Optional override path to a DCD trajectory file. If provided,
                this file is used instead of discovering coordinates from the
                parsed TeraChem input or scratch directory.
            rst7_path: Optional override path to an Amber restart file. If provided,
                this file is used instead of discovering coordinates from the
                parsed TeraChem input or scratch directory.
            prmtop_path: Optional override path to an Amber topology file. Used
                when the selected coordinate file is an Amber restart or DCD trajectory.
            qmindices_path: Optional override path to a TeraChem QM atom index
                file. Used when the selected coordinate file is an Amber restart
                or DCD trajectory.
        """

        if dcd_path is not None and rst7_path is not None:
            raise ValueError("Provide either dcd_path or rst7_path, not both.")

        # parse the stdout
        parsed = parse_terachem_output(file_path, raw_str_in=raw_str_in)
        inputs = parsed["input_args"]  # inputs to terachem
        excited_state_records = cls._terachem_records_with_frame_indices(
            parsed.get("excited_state_records", [])
        )

        stdout_dir = (
            os.getcwd()
            if raw_str_in
            else os.path.dirname(os.path.abspath(file_path)) or os.getcwd()
        )

        # determine runtype and location of crds
        scrdir = cls._resolve_terachem_scrdir(
            inputs["scrdir"],
            stdout_dir,
            None if raw_str_in else file_path,
        )
        runtype = inputs["run"]

        if dcd_path is not None:
            structure_path = cls._resolve_terachem_path(
                dcd_path,
                stdout_dir,
                scrdir=scrdir,
            )
        elif rst7_path is not None:
            structure_path = cls._resolve_terachem_path(
                rst7_path,
                stdout_dir,
                scrdir=scrdir,
            )
        elif runtype in ("energy", "gradient"):
            structure_path = cls._resolve_terachem_path(
                inputs["coordinates"],
                stdout_dir,
                scrdir=scrdir,
            )
        elif runtype in ("minimize", "conical"):
            optim_dcd_path = os.path.join(scrdir, "optim.dcd")
            optim_xyz_path = os.path.join(scrdir, "optim.xyz")
            optim_rst7_path = os.path.join(scrdir, "optim.rst7")
            has_optim_dcd = os.path.exists(optim_dcd_path)
            has_optim_xyz = os.path.exists(optim_xyz_path)
            has_optim_rst7 = os.path.exists(optim_rst7_path)

            if has_optim_dcd:
                structure_path = optim_dcd_path
            elif has_optim_xyz and has_optim_rst7:
                raise ValueError(
                    "Ambiguous TeraChem optimization structures: found both "
                    f"{optim_xyz_path} and {optim_rst7_path}."
                )
            elif has_optim_rst7:
                structure_path = optim_rst7_path
            else:
                structure_path = optim_xyz_path
        else:
            raise ValueError(
                f"Unknown TeraChem runtype '{runtype}' in parsed stdout. Add handling in Molecule() if needed."
            )

        if not os.path.exists(structure_path):
            raise FileNotFoundError(
                f"Expected TeraChem structure file not found at {structure_path}."
            )

        # NOTE: for now, i guess metadata will just have the input args?
        # seems slightly wasteful, unsure
        metadata = {"terachem_input_args": inputs}

        structure_ext = os.path.splitext(structure_path)[1].lower()
        if structure_ext in (".rst7", ".dcd"):
            if prmtop_path is None and inputs.get("prmtop") is None:
                raise ValueError(
                    "A prmtop path is required to parse Amber coordinate files."
                )
            resolved_prmtop_path = cls._resolve_terachem_path(
                prmtop_path if prmtop_path is not None else inputs["prmtop"],
                stdout_dir,
                scrdir=scrdir,
            )
            resolved_qmindices_path = None
            if qmindices_path is not None:
                resolved_qmindices_path = cls._resolve_terachem_path(
                    qmindices_path,
                    stdout_dir,
                    scrdir=scrdir,
                )
            elif inputs.get("qmindices") is not None:
                resolved_qmindices_path = cls._resolve_terachem_path(
                    inputs["qmindices"],
                    stdout_dir,
                    scrdir=scrdir,
                )
            if structure_ext == ".dcd":
                out = cls.from_dcd(
                    structure_path,
                    resolved_prmtop_path,
                    qmindices_path=resolved_qmindices_path,
                )
            else:
                out = cls.from_rst7(
                    structure_path,
                    resolved_prmtop_path,
                    qmindices_path=resolved_qmindices_path,
                )
            out.metadata.update(metadata)
            out.excited_state_records = excited_state_records
        else:
            xyz_payload = xyz_io.parse_xyz(structure_path)
            out = cls(
                _legacy_view="xyz",
                **xyz_payload,
                metadata=metadata,
                excited_state_records=excited_state_records,
            )

        return out

    @staticmethod
    def _resolve_terachem_path(
        path: str,
        stdout_dir: str,
        scrdir: str | None = None,
    ) -> str:
        """Resolve a TeraChem input path against common run directories.

        Args:
            path: Raw path value from the parsed TeraChem input.
            stdout_dir: Directory containing the stdout file.
            scrdir: Optional resolved TeraChem scratch directory.
        """
        if os.path.isabs(path):
            return path

        candidates = [os.path.abspath(os.path.join(stdout_dir, path))]
        if scrdir is not None:
            candidates.append(os.path.abspath(os.path.join(scrdir, path)))
        candidates.append(os.path.abspath(path))

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return candidates[0]

    @classmethod
    def _resolve_terachem_scrdir(
        cls,
        scrdir: str,
        stdout_dir: str,
        stdout_path: str | None = None,
    ) -> str:
        """Resolve a TeraChem scratch directory with stdout-tag fallback.

        Args:
            scrdir: Raw scrdir value from the parsed TeraChem input.
            stdout_dir: Directory containing the stdout file.
            stdout_path: Optional stdout file path. If its basename is
                ``stdout_<tag>.log``, ``tc_scr.<tag>`` beside the stdout file is
                tried when the parsed scrdir path does not exist.
        """
        resolved_scrdir = cls._resolve_terachem_path(scrdir, stdout_dir)
        if os.path.isdir(resolved_scrdir):
            return resolved_scrdir

        tag = cls._terachem_stdout_tag(stdout_path)
        if tag is not None:
            candidate = os.path.abspath(os.path.join(stdout_dir, f"tc_scr.{tag}"))
            if os.path.isdir(candidate):
                return candidate

        return resolved_scrdir

    @staticmethod
    def _terachem_stdout_tag(stdout_path: str | None) -> str | None:
        """Return the tag from a ``stdout_<tag>.log`` filename.

        Args:
            stdout_path: Optional stdout file path.
        """
        if stdout_path is None:
            return None

        basename = os.path.basename(stdout_path)
        prefix = "stdout_"
        suffix = ".log"
        if not basename.startswith(prefix) or not basename.endswith(suffix):
            return None

        tag = basename[len(prefix) : -len(suffix)]
        return tag or None
