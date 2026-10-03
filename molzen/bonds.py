"""Explicit connectivity and conservative distance-based bond suggestions."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np


BOND_DTYPE = np.dtype([("a", "i8"), ("b", "i8"), ("order", "U8"), ("source", "U12")])
BOND_ORDERS = (
    "unknown",
    "1",
    "2",
    "3",
    "4",
    "ar",
    "am",
    "du",
    "un",
    "nc",
    "dative",
    "complex",
    "ionic",
)
BOND_SOURCES = ("imported", "manual", "inferred", "template")
# Single-bond covalent radii in angstroms (Cordero et al., Dalton Trans., 2008,
# DOI: 10.1039/B801115J). Unsupported elements require explicit connectivity.
COVALENT_RADII = {
    "H": 0.31,
    "B": 0.84,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Si": 1.11,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Se": 1.20,
    "Br": 1.20,
    "I": 1.39,
}


class BondGraph:
    """An undirected graph whose endpoints are stable ``atom_index`` values.

    Args:
        records: Rows containing atom IDs, order, and source. Unknown order is
            written as ``"unknown"``; an absent graph is represented by None.
        status: ``"partial"`` or ``"complete"`` connectivity. This describes
            coverage, independently of whether individual orders are known.
    """

    def __init__(self, records: Iterable = (), *, status: str = "partial") -> None:
        if status not in ("partial", "complete"):
            raise ValueError("Bond status must be 'partial' or 'complete'.")
        rows = []
        seen = set()
        for row in records:
            a, b, order, source = row
            if any(
                isinstance(i, (bool, np.bool_))
                or not isinstance(i, (int, np.integer))
                or i < 0
                for i in (a, b)
            ):
                raise ValueError("Bond endpoints must be nonnegative integer atom IDs.")
            a, b = sorted((int(a), int(b)))
            order = str(order)
            if a == b or (a, b) in seen:
                raise ValueError("Self bonds and duplicate bonds are not allowed.")
            if order not in BOND_ORDERS or source not in BOND_SOURCES:
                raise ValueError("Unsupported bond order or source.")
            seen.add((a, b))
            rows.append((a, b, order, source))
        self._records = np.array(sorted(rows), dtype=BOND_DTYPE)
        self._status = status

    @property
    def records(self) -> np.ndarray:
        """Return a copy of the edge table."""
        return self._records.copy()

    @property
    def status(self) -> str:
        """Return the declared connectivity coverage."""
        return self._status

    def __len__(self) -> int:
        return len(self._records)

    def validate(self, atom_ids: Iterable[int]) -> None:
        """Check that atom IDs are unique and every endpoint exists."""
        ids = list(atom_ids)
        if len(set(ids)) != len(ids) or any(i < 0 for i in ids):
            raise ValueError(
                "Bond graphs require unique, nonnegative atom_index values."
            )
        if not set(self._records["a"]).union(self._records["b"]).issubset(ids):
            raise ValueError("Bond endpoint does not exist in atom_records.")

    def neighbors(self, atom_id: int) -> list[int]:
        """Return the atom IDs connected to an atom."""
        edges = self._records
        return sorted(
            [
                *edges["b"][edges["a"] == atom_id].tolist(),
                *edges["a"][edges["b"] == atom_id].tolist(),
            ]
        )

    def without_atoms(self, atom_ids: Iterable[int]) -> BondGraph:
        """Return a graph with the given atoms and their edges removed."""
        ids = list(atom_ids)
        keep = ~np.isin(self._records["a"], ids) & ~np.isin(self._records["b"], ids)
        return BondGraph(self._records[keep], status=self.status)

    def to_dict(self) -> dict:
        """Return a plain payload suitable for molecule serialization."""
        return {"records": self._records.tolist(), "status": self.status}


@dataclass
class BondSuggestions:
    """Candidate edges and diagnostics; accepting them is a separate operation."""

    bonds: BondGraph
    diagnostics: list[str]


def infer_bonds(
    atom_records: np.ndarray,
    *,
    frame: int = 0,
    existing: BondGraph | None = None,
    tolerance: float = 0.35,
) -> BondSuggestions:
    """Suggest connectivity using spatial bins and covalent radii.

    Args:
        atom_records: Canonical atom records with coordinates in angstroms.
        frame: Coordinate frame to examine.
        existing: Edges to preserve and exclude from the suggestions.
        tolerance: Additive distance tolerance in angstroms.

    Returns:
        New candidate edges with unknown order and diagnostic messages. No
        molecule is modified, and a partial graph never becomes complete here.
    """
    if not np.isfinite(tolerance) or not 0 <= tolerance <= 1:
        raise ValueError("tolerance must be between 0 and 1 angstrom.")
    if not 0 <= frame < atom_records.dtype["coords"].shape[0]:
        raise IndexError("Coordinate frame is out of range.")
    ids = atom_records["atom_index"]
    (existing if existing is not None else BondGraph()).validate(ids)
    xyz = atom_records["coords"][:, frame].astype(float)
    radii = np.array(
        [COVALENT_RADII.get(str(e), np.nan) for e in atom_records["element"]]
    )
    diagnostics = []
    unsupported = sorted(set(atom_records["element"][~np.isfinite(radii)]))
    if unsupported:
        diagnostics.append(f"No inference for elements: {', '.join(unsupported)}.")
    valid = np.isfinite(xyz).all(axis=1) & np.isfinite(radii)
    if not np.isfinite(xyz).all():
        diagnostics.append("Atoms with nonfinite coordinates were skipped.")
    cell_width = 2 * max(COVALENT_RADII.values()) + tolerance
    bins: dict[tuple, list[int]] = {}
    rows = []
    known = (
        set()
        if existing is None
        else {(int(r["a"]), int(r["b"])) for r in existing.records}
    )
    for i in np.flatnonzero(valid):
        cell = tuple(np.floor(xyz[i] / cell_width).astype(int))
        for offset in product((-1, 0, 1), repeat=3):
            key = tuple(c + d for c, d in zip(cell, offset))
            for j in bins.get(key, ()):
                a, b = sorted((int(ids[i]), int(ids[j])))
                if (a, b) in known:
                    continue
                # Different nonblank alternate locations are incompatible.
                alt_i, alt_j = atom_records["alt_loc"][[i, j]]
                if alt_i and alt_j and alt_i != alt_j:
                    continue
                distance = float(np.linalg.norm(xyz[i] - xyz[j]))
                radius_sum = radii[i] + radii[j]
                if distance < 0.55 * radius_sum:
                    diagnostics.append(
                        f"Atoms {a} and {b} are unusually close ({distance:.2f} A)."
                    )
                elif distance <= radius_sum + tolerance:
                    rows.append((a, b, "unknown", "inferred"))
        bins.setdefault(cell, []).append(i)
    suggested = BondGraph(rows)
    combined = BondGraph(
        [*(existing.records.tolist() if existing is not None else []), *rows]
    )
    limits = {"H": 1, "C": 4, "N": 4, "O": 2, "F": 1, "Cl": 1, "Br": 1, "I": 1}
    endpoints = np.concatenate([combined.records["a"], combined.records["b"]])
    unique_ids, counts = np.unique(endpoints, return_counts=True)
    degrees = dict(zip(unique_ids, counts, strict=True))
    for atom in atom_records:
        limit = limits.get(str(atom["element"]))
        if limit is not None and degrees.get(int(atom["atom_index"]), 0) > limit:
            diagnostics.append(
                f"Atom {atom['atom_index']} has unusually many neighbors; review its bonds."
            )
    return BondSuggestions(suggested, diagnostics)
