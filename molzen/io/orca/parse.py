"""Read geometries and energy/gradient results printed in ORCA stdout."""

from __future__ import annotations

import os
import re
from typing import Any

import numpy as np

from molzen.ptable import symbol_to_z


def _number(value: str) -> float:
    """Accept Fortran D exponents as well as ordinary decimal/E notation."""
    return float(value.replace("D", "E").replace("d", "e"))


def _read_vectors(
    lines: list[str], start: int, *, gradient: bool = False
) -> tuple[list[str], np.ndarray, int]:
    """Read a coordinate or indexed gradient table, rejecting malformed rows.

    start points to the line immediately after the table heading. Return the
    element labels, an (n_atoms, 3) array, and the first unread line index so
    the outer parser can resume scanning after the table.
    """
    elements = []
    vectors = []
    i = start
    # Headers are followed by dashed rules and, sometimes, blank lines.
    while i < len(lines) and (not lines[i].strip() or set(lines[i].strip()) == {"-"}):
        i += 1
    while i < len(lines):
        line = lines[i].strip()
        # Once rows have started, a blank line or dashed rule ends the table.
        if not line or set(line) == {"-"}:
            break
        parts = line.split()
        try:
            if gradient:
                # Gradient row:  1 N : dE/dX dE/dY dE/dZ
                if len(parts) != 6 or parts[2] != ":":
                    raise ValueError
                int(parts[0])
                element = parts[1].capitalize()
            else:
                # Coordinate row:  N X Y Z  (no atom index or colon).
                if len(parts) != 4:
                    raise ValueError
                element = parts[0].capitalize()
            if element not in symbol_to_z:
                raise ValueError
            vector = [_number(value) for value in parts[-3:]]
            # Reject NaN/Inf even though float() accepts them: these cannot
            # describe usable coordinates or derivatives.
            if not np.all(np.isfinite(vector)):
                raise ValueError
        except ValueError as exc:
            kind = "gradient" if gradient else "coordinate"
            raise ValueError(
                f"Invalid ORCA {kind} row at line {i + 1}: {line}"
            ) from exc
        elements.append(element)
        vectors.append(vector)
        i += 1
    if not vectors:
        raise ValueError(f"Empty ORCA Cartesian table near line {start + 1}.")
    return elements, np.asarray(vectors, dtype=float), i


def parse_orca_output(file_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Parse Angstrom coordinate blocks and their subsequent energy/gradient data.

    Every printed Angstrom geometry becomes a frame (including repeated final
    geometries). Results belong to the most recently printed frame. Atomic-unit
    coordinate tables are ignored. No companion files are needed.

    The returned dictionary can be passed to Molecule's constructor. Energies
    and gradients share one record per frame in excited_state_records; this
    is the container's existing results field, even for ground-state data.
    """
    with open(file_path, encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()

    frames = []
    elements = None
    # Keying results by frame lets separately printed energies and gradients
    # populate the same record. Frames without results have no record.
    records: dict[int, dict[str, Any]] = {}
    charge = None
    spinmult = None
    atom_counts = set()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # ORCA also prints the same geometry in atomic units. Read only the
        # Angstrom table so it does not become a duplicate, mis-scaled frame.
        if line in (
            "CARTESIAN COORDINATES (ANGSTROEM)",
            "CARTESIAN COORDINATES (ANGSTROM)",
        ):
            labels, coords, i = _read_vectors(lines, i + 1)
            # Frames must describe the same ordered atoms to form a trajectory.
            if elements is not None and labels != elements:
                raise ValueError(
                    "ORCA coordinate blocks have inconsistent atom identities/order."
                )
            elements = labels
            frames.append(coords)
            continue

        # These summary fields may occur more than once. Repeated identical
        # values are fine, but conflicting values cannot describe one Molecule.
        # The regexes tolerate ORCA's variable spacing and dotted separators.
        match = re.match(r"Number of atoms\s+\.+\s+(\d+)\s*$", line)
        if match:
            atom_counts.add(int(match[1]))
        match = re.match(r"Total Charge\s+Charge\s+\.+\s+([+-]?\d+)\s*$", line)
        if match:
            value = int(match[1])
            if charge is not None and charge != value:
                raise ValueError("ORCA output contains different molecular charges.")
            charge = value
        # Multiplicity appears as either "Multiplicity Mult .... 1" or
        # "Multiplicity : 1", depending on the output section.
        match = re.match(r"Multiplicity\s+(?:Mult\s+\.+|:)\s+(\d+)\s*$", line)
        if match:
            value = int(match[1])
            if spinmult is not None and spinmult != value:
                raise ValueError("ORCA output contains different spin multiplicities.")
            spinmult = value

        is_energy = line.startswith("FINAL SINGLE POINT ENERGY")
        is_gradient = line == "CARTESIAN GRADIENT"
        if is_energy or is_gradient:
            if not frames:
                raise ValueError(
                    "ORCA energy/gradient appears before Angstrom coordinates."
                )
            # Associate results with the latest geometry seen in file order.
            # A repeated result for that geometry replaces the earlier value.
            frame_index = len(frames) - 1
            record = records.setdefault(
                frame_index,
                {"source": "parse_orca_output", "frame_index": frame_index},
            )
            if is_energy:
                try:
                    # FINAL SINGLE POINT ENERGY <value in Hartree>
                    energy = _number(line.split()[4])
                    if not np.isfinite(energy):
                        raise ValueError
                except (IndexError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid ORCA final energy at line {i + 1}."
                    ) from exc
                record["total_energy_au"] = energy
            else:
                labels, gradient, i = _read_vectors(lines, i + 1, gradient=True)
                # Check both row count and element order before attaching
                # atom-wise derivatives to this frame.
                if labels != elements:
                    raise ValueError(
                        "ORCA gradient atoms do not match the coordinates."
                    )
                record["energy_gradient"] = gradient.tolist()
                continue
        i += 1

    if not frames:
        raise ValueError("No Cartesian coordinates in Angstrom found in ORCA stdout.")
    # Atom-count summaries can occur after the coordinates, so validate them
    # after scanning the whole file. This also catches truncated geometries.
    if atom_counts and atom_counts != {len(elements)}:
        raise ValueError("ORCA coordinate atom count does not match Number of atoms.")
    # Keep coordinates frame-major here; Molecule handles its public xyz shape.
    # Termination is metadata rather than a requirement, so partial jobs with
    # usable geometries can still be inspected.
    return {
        "xyz": np.stack(frames),
        "elements": elements,
        "spinmult": spinmult,
        "excited_state_records": list(records.values()),
        "metadata": {
            "orca": {
                "charge": charge,
                "terminated_normally": any(
                    "ORCA TERMINATED NORMALLY" in line for line in lines
                ),
                "coordinate_units": "angstrom",
                "energy_units": "hartree",
                "gradient_units": "hartree/bohr",
            }
        },
    }
