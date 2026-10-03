"""MOL2 format readers/writers that operate on plain data."""

from __future__ import annotations

from typing import Any

import numpy as np

from molzen.ptable import ALL_SYMBOLS
from molzen.bonds import BondGraph

MOL2_HETATM_DTYPES = np.dtype(
    [
        ("atom_idx", "i4"),
        ("atom_name", "U8"),
        ("atom_type", "U8"),
        ("element", "U4"),
        ("res_name", "U3"),
        ("xyz", "f4", (3,)),
    ]
)

_ELEMENT_SYMBOLS = set(ALL_SYMBOLS)


def _infer_element_from_atom_name(atom_name: str) -> str:
    letters = "".join(ch for ch in atom_name if ch.isalpha())
    if not letters:
        return ""
    if len(letters) >= 2:
        cand2 = letters[:2].capitalize()
        if cand2 in _ELEMENT_SYMBOLS:
            return cand2
    cand1 = letters[0].upper()
    if cand1 in _ELEMENT_SYMBOLS:
        return cand1
    return ""


def parse_mol2(mol2_fp: str) -> dict[str, Any]:
    """Parse a `.mol2` file and extract atom-level data."""
    hetatm_data = []
    in_atom_section = False
    saw_atom_section = False

    with open(mol2_fp, "r") as f:
        lines = f.readlines()

    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue
        if line.startswith("@<TRIPOS>ATOM"):
            in_atom_section = True
            saw_atom_section = True
            continue
        if line.startswith("@<TRIPOS>"):
            in_atom_section = False
            continue
        if not in_atom_section:
            continue

        split = line.split()
        if len(split) < 6:
            raise ValueError(
                f"Malformed MOL2 atom line {line_no}: expected >= 6 columns."
            )

        try:
            atom_idx = int(split[0])
            atom_name = split[1]
            x = float(split[2])
            y = float(split[3])
            z = float(split[4])
            atom_type = split[5]
        except ValueError as exc:
            raise ValueError(f"Invalid MOL2 atom line {line_no}.") from exc

        element = atom_type.split(".")[0].capitalize()
        if element not in _ELEMENT_SYMBOLS:
            element = _infer_element_from_atom_name(atom_name)
        res_name = split[7] if len(split) > 7 else ""

        het = np.array(
            (atom_idx, atom_name, atom_type, element, res_name, np.array([x, y, z])),
            dtype=MOL2_HETATM_DTYPES,
        )
        hetatm_data.append(het)

    if not saw_atom_section:
        raise ValueError(f"No @<TRIPOS>ATOM section found in MOL2 file: {mol2_fp}")

    hetatm = np.array(hetatm_data, dtype=MOL2_HETATM_DTYPES)
    xyz = hetatm["xyz"] if hetatm.size else np.empty((0, 3), dtype=float)
    atom_names = hetatm["atom_name"].tolist() if hetatm.size else []
    elements = hetatm["element"].tolist() if hetatm.size else []

    atom_map = {int(row["atom_idx"]): i for i, row in enumerate(hetatm)}
    if len(atom_map) != len(hetatm):
        raise ValueError("MOL2 atom IDs must be unique.")
    bond_rows = []
    in_bonds = False
    saw_bonds = False
    for line in lines:
        if line.startswith("@<TRIPOS>"):
            in_bonds = line.strip() == "@<TRIPOS>BOND"
            saw_bonds |= in_bonds
        elif in_bonds and line.strip() and not line.lstrip().startswith("#"):
            fields = line.split()
            try:
                bond_rows.append(
                    (
                        atom_map[int(fields[1])],
                        atom_map[int(fields[2])],
                        fields[3],
                        "imported",
                    )
                )
            except (IndexError, KeyError, ValueError) as exc:
                raise ValueError(f"Invalid MOL2 bond record: {line.strip()}") from exc

    return {
        "xyz": xyz,
        "atom_names": atom_names,
        "elements": elements,
        "hetatm": hetatm,
        "bonds": BondGraph(
            bond_rows,
            status="partial"
            if "# MOLZEN_BOND_STATUS partial\n" in lines
            else "complete",
        )
        if saw_bonds
        else None,
    }


def write_mol2(
    filename: str,
    xyz: np.ndarray | None,
    atom_names: list[str] | None = None,
    elements: list[str] | None = None,
    hetatm: np.ndarray | None = None,
    return_str: bool = False,
    *,
    bonds: BondGraph | None = None,
    atom_ids: list[int] | None = None,
) -> str | None:
    """Write MOL2 text to disk or return it as a string."""
    atom_lines: list[str] = []

    if hetatm is not None and len(hetatm) > 0:
        dtype_names = set(hetatm.dtype.names or ())
        required = {"atom_idx", "atom_name", "atom_type", "element", "res_name", "xyz"}
        if not required.issubset(dtype_names):
            missing = sorted(required - dtype_names)
            raise ValueError(f"MOL2 hetatm array is missing required fields: {missing}")

        for row in hetatm:
            atom_idx = int(row["atom_idx"])
            atom_name = str(row["atom_name"]).strip() or f"{row['element']}{atom_idx}"
            atom_type = str(row["atom_type"])
            element = str(row["element"])
            res_name = str(row["res_name"]) or "MOL"
            coord = np.asarray(row["xyz"], dtype=float)
            atom_lines.append(
                f"{atom_idx:7d} {atom_name:<8} {coord[0]:10.4f} {coord[1]:10.4f} "
                f"{coord[2]:10.4f} {atom_type:<8} 1 {res_name:<3} 0.0000\n"
            )
            if not element:
                element = _infer_element_from_atom_name(atom_name)
    else:
        if xyz is None:
            raise ValueError("xyz is required when hetatm is not provided.")
        xyz = np.asarray(xyz, dtype=float)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("xyz must have shape (N, 3) for MOL2 output.")

        natoms = xyz.shape[0]
        if atom_names is None:
            atom_names = [f"A{i + 1}" for i in range(natoms)]
        if elements is None:
            elements = ["X"] * natoms
        if len(atom_names) != natoms or len(elements) != natoms:
            raise ValueError("atom_names and elements must match xyz atom count.")

        for i in range(natoms):
            atom_idx = i + 1
            atom_name = atom_names[i]
            element = elements[i]
            atom_type = element if element else "X"
            coord = xyz[i]
            atom_lines.append(
                f"{atom_idx:7d} {atom_name:<8} {coord[0]:10.4f} {coord[1]:10.4f} "
                f"{coord[2]:10.4f} {atom_type:<8} 1 MOL 0.0000\n"
            )

    natoms = len(atom_lines)
    bond_lines = []
    if bonds is not None:
        ids = list(range(natoms)) if atom_ids is None else atom_ids
        if len(ids) != natoms:
            raise ValueError("atom_ids must match the atom count.")
        bonds.validate(ids)
        serials = [int(line.split()[0]) for line in atom_lines]
        if len(set(serials)) != natoms:
            raise ValueError("MOL2 output atom IDs must be unique.")
        serial_map = dict(zip(ids, serials, strict=True))
        for i, edge in enumerate(bonds.records, start=1):
            if edge["order"] in ("4", "dative", "complex", "ionic"):
                raise ValueError(
                    f"MOL2 cannot represent bond type {edge['order']!r}; use NPY or HDF5."
                )
            order = "un" if edge["order"] == "unknown" else str(edge["order"])
            bond_lines.append(
                f"{i:6d} {serial_map[int(edge['a'])]:6d} {serial_map[int(edge['b'])]:6d} {order}\n"
            )
    out = [
        "@<TRIPOS>MOLECULE\n",
        "MOLZEN\n",
        f"{natoms} {len(bond_lines)} 0 0 0\n",
        "SMALL\n",
        "NO_CHARGES\n",
        "\n",
        "@<TRIPOS>ATOM\n",
        *atom_lines,
    ]
    if bonds is not None:
        out.extend(
            [f"# MOLZEN_BOND_STATUS {bonds.status}\n", "@<TRIPOS>BOND\n", *bond_lines]
        )
    outstr = "".join(out)

    if return_str:
        return outstr

    with open(filename, "w") as f:
        f.write(outstr)
    return None
