"""Transactional molecule edits independent of notebook and viewer packages."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np

from molzen.bonds import BondGraph, BondSuggestions, COVALENT_RADII
from molzen.ptable import symbol_to_z

if TYPE_CHECKING:
    from molzen.io.molecule import Molecule


class EditSession:
    """Edit a single-frame copy with undo and redo support.

    Args:
        molecule: Source molecule. Its records and metadata are copied.
        frame: Frame to edit; required when the source has multiple frames.
        history_limit: Maximum number of undo snapshots to retain.
    """

    def __init__(
        self, molecule: Molecule, *, frame: int | None = None, history_limit: int = 50
    ) -> None:
        records = molecule.atom_records
        if molecule.metadata.get("pdb_model_count", 0) > 1:
            raise ValueError("Select one PDB MODEL before opening the editor.")
        if records is None or not len(records):
            raise ValueError("Cannot edit an empty molecule.")
        if frame is None and records.dtype["coords"].shape[0] != 1:
            raise ValueError("Choose a frame explicitly when editing a trajectory.")
        if history_limit < 1:
            raise ValueError("history_limit must be positive.")
        self._molecule = deepcopy(molecule[0 if frame is None else frame])
        BondGraph().validate(self._molecule.atom_records["atom_index"])
        if not np.isfinite(self._molecule.atom_records["coords"]).all():
            raise ValueError("Editing requires finite coordinates.")
        self._undo: list[Molecule] = []
        self._redo: list[Molecule] = []
        self._history_limit = history_limit
        self.revision = 0
        self.suggestions: BondSuggestions | None = None

    @property
    def molecule(self) -> Molecule:
        """Return an independent copy of the latest accepted structure."""
        return deepcopy(self._molecule)

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def suggest_bonds(self, *, tolerance: float = 0.35) -> BondSuggestions:
        """Preview distance-based candidates without changing connectivity."""
        self.suggestions = self._molecule.infer_bonds(tolerance=tolerance)
        return self.suggestions

    def apply(self, action: str, **args: Any) -> None:
        """Apply one validated operation and record an undo snapshot.

        Args:
            action: One of set_element, delete_atom, set_bond, remove_bond,
                replace_hydrogen, or accept_bonds.
            **args: Operation arguments, using stable atom IDs for endpoints.

        Raises:
            ValueError: If the operation or its chemistry constraints are invalid.
        """
        operations = {
            "set_element": _set_element,
            "delete_atom": _delete_atom,
            "set_bond": _set_bond,
            "remove_bond": _remove_bond,
            "replace_hydrogen": _replace_hydrogen,
        }
        candidate = self.molecule
        if action == "accept_bonds":
            if self.suggestions is None:
                raise ValueError("Preview bond suggestions before accepting them.")
            current = (
                [] if candidate.bonds is None else candidate.bonds.records.tolist()
            )
            candidate.bonds = BondGraph(
                [*current, *self.suggestions.bonds.records.tolist()]
            )
        elif action in operations:
            operations[action](candidate, **args)
        else:
            raise ValueError(f"Unknown edit operation: {action}.")
        if candidate.bonds is not None:
            candidate.bonds.validate(candidate.atom_records["atom_index"])
        # Preserve source metadata as provenance, not as properties of the edit.
        if not candidate.metadata.get("molzen_edit"):
            candidate.metadata = {
                "molzen_edit": True,
                "source_metadata": deepcopy(candidate.metadata),
                "source_spinmult": candidate.spinmult,
                "source_comments": candidate.comments,
            }
        candidate.spinmult = None
        candidate.excited_state_records = None
        candidate.comments = ["Edited molecule"]
        candidate._clear_stale_pdb_metadata()
        self._undo.append(self._molecule)
        self._undo = self._undo[-self._history_limit :]
        self._redo.clear()
        self._molecule = candidate
        self.suggestions = None
        self.revision += 1

    def undo(self) -> None:
        """Restore the previous structure, including topology and metadata."""
        if not self._undo:
            raise ValueError("Nothing to undo.")
        self._redo.append(self._molecule)
        self._molecule = self._undo.pop()
        self.suggestions = None
        self.revision += 1

    def redo(self) -> None:
        """Restore the most recently undone structure."""
        if not self._redo:
            raise ValueError("Nothing to redo.")
        self._undo.append(self._molecule)
        self._molecule = self._redo.pop()
        self.suggestions = None
        self.revision += 1


def _atom_position(molecule: Molecule, atom_id: int) -> int:
    """Resolve a stable atom ID to its current row position."""
    if isinstance(atom_id, bool) or not isinstance(atom_id, (int, np.integer)):
        raise ValueError("atom_id must be an integer.")
    matches = np.flatnonzero(molecule.atom_records["atom_index"] == atom_id)
    if len(matches) != 1:
        raise ValueError(f"Atom {atom_id} does not exist or is not unique.")
    return int(matches[0])


def _set_element(molecule: Molecule, *, atom_id: int, element: str) -> None:
    """Change an element and discard its old type and charge annotations."""
    if element not in symbol_to_z:
        raise ValueError(f"Unknown element: {element}.")
    position = _atom_position(molecule, atom_id)
    row = molecule.atom_records[position]
    row["element"] = element
    row["atom_type"] = ""
    row["charge"] = ""
    row["atom_name"] = f"{element}{atom_id}"


def _delete_atom(molecule: Molecule, *, atom_id: int) -> None:
    """Remove one atom and all incident edges."""
    if len(molecule.atom_records) == 1:
        raise ValueError("Keep at least one atom in the editor.")
    molecule.pop(_atom_position(molecule, atom_id))


def _set_bond(molecule: Molecule, *, a: int, b: int, order: str = "1") -> None:
    """Add a bond or replace its order, recording a manual assignment."""
    _atom_position(molecule, a)
    _atom_position(molecule, b)
    a, b = sorted((a, b))
    old = molecule.bonds
    rows = (
        []
        if old is None
        else [tuple(r) for r in old.records if (r["a"], r["b"]) != (a, b)]
    )
    molecule.bonds = BondGraph(
        [*rows, (a, b, order, "manual")],
        status="partial" if old is None else old.status,
    )


def _remove_bond(molecule: Molecule, *, a: int, b: int) -> None:
    """Remove an explicitly stored bond."""
    _atom_position(molecule, a)
    _atom_position(molecule, b)
    a, b = sorted((a, b))
    if molecule.bonds is None:
        raise ValueError("No bond graph is available.")
    rows = [tuple(r) for r in molecule.bonds.records if (r["a"], r["b"]) != (a, b)]
    if len(rows) == len(molecule.bonds):
        raise ValueError("The selected atoms are not bonded.")
    molecule.bonds = BondGraph(rows, status=molecule.bonds.status)


def _replace_hydrogen(
    molecule: Molecule,
    *,
    atom_id: int,
    torsion: float = 0.0,
    bond_length: float | None = None,
) -> None:
    """Replace a terminal hydrogen with an idealized methyl group.

    Args:
        molecule: Working molecule to modify.
        atom_id: Hydrogen with one confirmed single bond to C, N, O, or S.
        torsion: Methyl rotation in degrees around the attachment axis.
        bond_length: Optional parent-carbon distance in angstroms.
    """
    position = _atom_position(molecule, atom_id)
    records = molecule.atom_records
    if records[position]["element"] != "H" or molecule.bonds is None:
        raise ValueError("Select a hydrogen with explicit connectivity.")
    graph = molecule.bonds
    neighbors = graph.neighbors(atom_id)
    if len(neighbors) != 1:
        raise ValueError("The hydrogen must have exactly one bonded parent.")
    edge = next(r for r in graph.records if atom_id in (r["a"], r["b"]))
    if edge["order"] != "1" or edge["source"] == "inferred":
        raise ValueError("Confirm the hydrogen-parent bond as a single bond first.")
    parent = records[_atom_position(molecule, neighbors[0])]
    lengths = {"C": 1.52, "N": 1.47, "O": 1.43, "S": 1.82}
    if parent["element"] not in lengths:
        raise ValueError("Methyl attachment currently supports C, N, O, and S parents.")
    length = (
        lengths[str(parent["element"])] if bond_length is None else float(bond_length)
    )
    if not np.isfinite(length) or not 0.5 <= length <= 3 or not np.isfinite(torsion):
        raise ValueError("Use a finite torsion and a bond length between 0.5 and 3 A.")
    axis = (records[position]["coords"][0] - parent["coords"][0]).astype(float)
    norm = np.linalg.norm(axis)
    if norm < 1e-6:
        raise ValueError("Hydrogen and parent coordinates coincide.")
    axis /= norm
    reference = np.eye(3)[np.argmin(np.abs(axis))]
    u = np.cross(axis, reference)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    carbon = parent["coords"][0] + length * axis
    records[position]["coords"][0] = carbon
    _set_element(molecule, atom_id=atom_id, element="C")
    added = np.repeat(records[position : position + 1], 3).copy()
    next_id = int(records["atom_index"].max()) + 1
    if next_id + 2 > np.iinfo(np.int32).max:
        raise ValueError("No atom IDs remain in the canonical dtype.")
    new_ids = list(range(next_id, next_id + 3))
    added["atom_index"] = new_ids
    added["serial"] = np.arange(
        int(records["serial"].max()) + 1, int(records["serial"].max()) + 4
    )
    added["element"] = "H"
    added["atom_name"] = [f"H{i}" for i in new_ids]
    for i, theta in enumerate(np.deg2rad(torsion) + np.arange(3) * 2 * np.pi / 3):
        direction = axis / 3 + np.sqrt(8 / 9) * (np.cos(theta) * u + np.sin(theta) * v)
        added[i]["coords"][0] = carbon + 1.09 * direction
    molecule.atom_records = np.concatenate([records, added])
    molecule.bonds = BondGraph(
        [*graph.records.tolist(), *((atom_id, i, "1", "template") for i in new_ids)],
        status=graph.status,
    )


def close_contacts(molecule: Molecule) -> list[str]:
    """Report short nonbonded contacts using the same spatial search as inference."""
    suggestions = molecule.infer_bonds(tolerance=0)
    messages = list(suggestions.diagnostics)
    by_id = {int(r["atom_index"]): r for r in molecule.atom_records}
    for edge in suggestions.bonds.records:
        a, b = by_id[int(edge["a"])], by_id[int(edge["b"])]
        distance = np.linalg.norm(a["coords"][0] - b["coords"][0])
        threshold = (
            COVALENT_RADII[str(a["element"])] + COVALENT_RADII[str(b["element"])]
        )
        if distance < 0.85 * threshold:
            messages.append(
                f"Close nonbonded contact: atoms {edge['a']} and {edge['b']} ({distance:.2f} A)."
            )
    return messages
