"""Lightweight standardized container for molecule data."""

from __future__ import annotations

from typing import Any, Iterator, Mapping

import numpy as np

from molzen.amino_acids import aa2long, aa2num, aa_1_to_3, ncaas, oneletter_code
from molzen.ptable import symbol_to_z
from . import hdf5 as hdf5_io
from . import mol2 as mol2_io
from . import npy as npy_io
from . import pdb as pdb_io
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
        "seq",
        "hetatm",
    )

    def __init__(
        self,
        xyz: np.ndarray | None = None,
        atom_names: list[str] | None = None,
        elements: list[str] | None = None,
        comments: list[str] | None = None,
        seq: str | None = None,
        hetatm: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
        atom_records: np.ndarray | None = None,
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
        self._metadata: dict[str, Any] = {}

        self.comments = comments
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
        self._atom_records = records

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

    def __getitem__(self, key: str) -> Any:
        if key == "metadata":
            if self.metadata:
                return self.metadata
            raise KeyError(key)

        if key not in self._MAPPING_FIELDS:
            raise KeyError(key)

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

        if self.atom_records is not None:
            parts.append(f"atom_records={len(self.atom_records)}")
            parts.append(f"frames={self._frame_count(self.atom_records)}")
        if self.xyz is not None:
            parts.append(f"xyz_shape={tuple(np.asarray(self.xyz).shape)}")
        if self.polymer_xyz is not None:
            parts.append(f"polymer_xyz_shape={tuple(self.polymer_xyz.shape)}")
        if self.atom_names is not None:
            parts.append(f"atom_names={len(self.atom_names)}")
        if self.elements is not None:
            parts.append(f"elements={len(self.elements)}")
        if self.comments is not None:
            parts.append(f"comments={len(self.comments)}")
        if self.seq is not None:
            parts.append(f"seq_len={len(self.seq)}")
        if self.hetatm is not None:
            parts.append(f"hetatm={len(self.hetatm)}")
        if self.metadata:
            parts.append(f"metadata_keys={sorted(self.metadata)}")

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
