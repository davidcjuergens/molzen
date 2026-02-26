"""Lightweight standardized container for molecule data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

import numpy as np

from . import hdf5 as hdf5_io
from . import mol2 as mol2_io
from . import npy as npy_io
from . import pdb as pdb_io
from . import xyz as xyz_io


@dataclass
class Molecule(Mapping[str, Any]):
    """Simple molecule container. Used by molzen.io parsers.

    Intentionally shallow and explicit so users can quickly inspect
    its fields.
    """

    xyz: np.ndarray | None = None
    atom_names: list[str] | None = None
    elements: list[str] | None = None
    comments: list[str] | None = None
    seq: str | None = None
    hetatm: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    _MAPPING_FIELDS = ("xyz", "atom_names", "elements", "comments", "seq", "hetatm")

    def __post_init__(self) -> None:
        if self.xyz is not None:
            xyz = np.asarray(self.xyz, dtype=float)
            if xyz.size and xyz.shape[-1] != 3:
                raise ValueError("xyz must have shape (..., 3).")
            self.xyz = xyz

        if self.comments is not None and self.xyz is not None and self.xyz.ndim == 3:
            if len(self.comments) != self.xyz.shape[0]:
                raise ValueError(
                    "comments length must match number of coordinate frames."
                )

    def _present_items(self) -> dict[str, Any]:
        data = {
            field: getattr(self, field)
            for field in self._MAPPING_FIELDS
            if getattr(self, field) is not None
        }
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

    def as_dict(self, include_none: bool = False) -> dict[str, Any]:
        """Return a dictionary representation of the molecule."""
        if include_none:
            data = {field: getattr(self, field) for field in self._MAPPING_FIELDS}
            data["metadata"] = self.metadata
            return data
        return self._present_items()

    def __repr__(self) -> str:
        parts: list[str] = []

        if self.xyz is not None:
            parts.append(f"xyz_shape={tuple(self.xyz.shape)}")
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

    @classmethod
    def from_xyz(cls, file_path: str) -> Molecule:
        """Load a molecule from an XYZ file path."""
        return cls(**xyz_io.parse_xyz(file_path))

    def to_xyz(self, file_path: str, return_str: bool = False) -> str | None:
        """Write this molecule to XYZ format."""
        if self.xyz is None:
            raise ValueError("xyz is required for XYZ output.")
        if self.elements is None:
            raise ValueError("elements are required for XYZ output.")
        return xyz_io.write_xyz(
            file_path,
            self.xyz,
            self.elements,
            comments=self.comments,
            return_str=return_str,
        )

    @classmethod
    def from_pdb(cls, file_path: str) -> Molecule:
        """Load a molecule from a PDB file path."""
        return cls(**pdb_io.parse_pdb(file_path))

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
        return pdb_io.write_pdb(
            file_path,
            xyz=self.xyz,
            seq=self.seq,
            chains=chains,
            hetatm=self.hetatm,
            pdb_raw_lines=raw_lines,
            return_str=return_str,
        )

    @classmethod
    def from_mol2(cls, file_path: str) -> Molecule:
        """Load a molecule from a MOL2 file path."""
        return cls(**mol2_io.parse_mol2(file_path))

    def to_mol2(self, file_path: str, return_str: bool = False) -> str | None:
        """Write this molecule to MOL2 format."""
        return mol2_io.write_mol2(
            file_path,
            xyz=self.xyz,
            atom_names=self.atom_names,
            elements=self.elements,
            hetatm=self.hetatm,
            return_str=return_str,
        )

    @classmethod
    def from_npy(cls, file_path: str) -> Molecule:
        """Load a molecule from an NPY file path."""
        return cls(**npy_io.parse_npy(file_path))

    def to_npy(self, file_path: str, return_bytes: bool = False) -> bytes | None:
        """Write this molecule to NPY format."""
        return npy_io.write_npy(
            file_path, self.as_dict(include_none=False), return_bytes
        )

    @classmethod
    def from_hdf5(cls, file_path: str) -> Molecule:
        """Load a molecule from an HDF5 file path."""
        return cls(**hdf5_io.parse_hdf5(file_path))

    def to_hdf5(self, file_path: str) -> None:
        """Write this molecule to HDF5 format."""
        hdf5_io.write_hdf5(file_path, self.as_dict(include_none=False))
