"""PDB format readers/writers that operate on plain data."""

from __future__ import annotations

from typing import Any

import numpy as np

from molzen.amino_acids import aa2long, aa2num, aa_1_to_3, ncaas, num2aa, oneletter_code
from molzen.ptable import ALL_SYMBOLS

HETATM_DTYPES = np.dtype(
    [
        ("atom_name", "U4"),
        ("res_name", "U3"),
        ("chain_id", "U1"),
        ("res_num", "i4"),
        ("xyz", "f4", (3,)),
    ]
)

PDB_RECORD_DTYPES = np.dtype(
    [
        ("record_name", "U6"),
        ("serial", "i4"),
        ("atom_name", "U4"),
        ("alt_loc", "U1"),
        ("res_name", "U3"),
        ("chain_id", "U1"),
        ("res_num", "i4"),
        ("i_code", "U1"),
        ("xyz", "f4", (3,)),
        ("occupancy", "f4"),
        ("temp_factor", "f4"),
        ("element", "U2"),
        ("charge", "U2"),
    ]
)

_ELEMENT_SYMBOLS = set(ALL_SYMBOLS)


def _parse_int_field(raw: str, field_name: str, line_no: int) -> int:
    text = raw.strip()
    if not text:
        raise ValueError(f"Missing {field_name} at PDB line {line_no}.")
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(
            f"Invalid {field_name} at PDB line {line_no}: {text!r}"
        ) from exc


def _parse_float_field(raw: str, field_name: str, line_no: int) -> float:
    text = raw.strip()
    if not text:
        raise ValueError(f"Missing {field_name} at PDB line {line_no}.")
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(
            f"Invalid {field_name} at PDB line {line_no}: {text!r}"
        ) from exc


def _parse_optional_float_field(raw: str, field_name: str, line_no: int) -> float:
    text = raw.strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(
            f"Invalid {field_name} at PDB line {line_no}: {text!r}"
        ) from exc


def _infer_element(atom_name: str, explicit_element: str) -> str:
    element = explicit_element.strip().capitalize()
    if element in _ELEMENT_SYMBOLS:
        return element

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


def parse_pdb(pdb_fp: str) -> dict[str, Any]:
    """Parse a `.pdb` file and extract residue-level, HETATM, and full record metadata."""
    aa_atom_idx = [
        {name.strip(): i for i, name in enumerate(long) if name is not None}
        for long in aa2long
    ]

    with open(pdb_fp, "r") as f:
        raw_lines = f.readlines()

    xyz = []
    seq = []
    hetatm_data = []
    pdb_records = []

    cur_res_id = None
    cur_xyz = None

    for line_no, line in enumerate(raw_lines, start=1):
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if len(line) < 54:
            raise ValueError(f"PDB ATOM/HETATM line {line_no} is too short.")

        record_name = line[0:6].strip()
        serial = _parse_int_field(line[6:11], "atom serial", line_no)
        atom_name = line[12:16].strip()
        alt_loc = line[16].strip() if len(line) > 16 else ""
        res_name = line[17:20].strip()
        chain_id = line[21].strip() if len(line) > 21 else ""
        res_num = _parse_int_field(line[22:26], "residue number", line_no)
        i_code = line[26].strip() if len(line) > 26 else ""
        x = _parse_float_field(line[30:38], "x coordinate", line_no)
        y = _parse_float_field(line[38:46], "y coordinate", line_no)
        z = _parse_float_field(line[46:54], "z coordinate", line_no)
        occupancy = _parse_optional_float_field(
            line[54:60] if len(line) >= 60 else "", "occupancy", line_no
        )
        temp_factor = _parse_optional_float_field(
            line[60:66] if len(line) >= 66 else "", "temperature factor", line_no
        )
        element = _infer_element(
            atom_name, line[76:78] if len(line) >= 78 else atom_name
        )
        charge = line[78:80].strip() if len(line) >= 80 else ""

        pdb_records.append(
            np.array(
                (
                    record_name,
                    serial,
                    atom_name,
                    alt_loc,
                    res_name,
                    chain_id,
                    res_num,
                    i_code,
                    np.array([x, y, z], dtype=float),
                    occupancy,
                    temp_factor,
                    element,
                    charge,
                ),
                dtype=PDB_RECORD_DTYPES,
            )
        )

        if record_name == "ATOM":
            res_id = (chain_id, res_num, i_code)
            if res_id != cur_res_id:
                if cur_res_id is not None:
                    xyz.append(cur_xyz)

                res_oneletter = oneletter_code.get(res_name)
                if res_oneletter is None:
                    ncaa = ncaas.get(res_name)
                    if ncaa is None or ncaa.get("canonical_one_letter") is None:
                        raise KeyError(f"Unknown residue name: {res_name}")
                    res_oneletter = ncaa["canonical_one_letter"]

                seq.append(res_oneletter)
                cur_res_id = res_id
                cur_xyz = np.full((27, 3), np.nan, dtype=float)

            aa_index = aa2num[res_name]
            atom_map = aa_atom_idx[aa_index]
            if atom_name in atom_map:
                atom_idx = atom_map[atom_name]
                cur_xyz[atom_idx] = np.array([x, y, z], dtype=float)

        elif record_name == "HETATM":
            het = np.array(
                (atom_name, res_name, chain_id, res_num, np.array([x, y, z])),
                dtype=HETATM_DTYPES,
            )
            hetatm_data.append(het)

    if cur_res_id is not None:
        xyz.append(cur_xyz)

    metadata = {
        "pdb_raw_lines": raw_lines,
        "pdb_records": np.array(pdb_records, dtype=PDB_RECORD_DTYPES),
    }

    return {
        "xyz": np.array(xyz, dtype=float),
        "seq": "".join(seq),
        "hetatm": np.array(hetatm_data, dtype=HETATM_DTYPES),
        "metadata": metadata,
    }


def write_pdb(
    file_path: str,
    xyz: np.ndarray | None,
    seq,
    chains: list[str] | None = None,
    hetatm: np.ndarray | None = None,
    *,
    pdb_raw_lines: list[str] | None = None,
    return_str: bool = False,
) -> str | None:
    """Write PDB text to disk or return it as a string."""
    if pdb_raw_lines is not None:
        outstr = "".join(
            line if line.endswith("\n") else f"{line}\n" for line in pdb_raw_lines
        )
        if return_str:
            return outstr
        with open(file_path, "w") as f:
            f.write(outstr)
        return None

    if xyz is None or seq is None:
        raise ValueError("xyz and seq are required when pdb_raw_lines is not provided.")

    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim != 3 or xyz.shape[2] != 3:
        raise ValueError("xyz must have shape (Nres, Natom, 3).")

    seq_values = list(seq) if isinstance(seq, str) else list(seq)
    nres = xyz.shape[0]
    if len(seq_values) != nres:
        raise ValueError("Sequence length must match number of residues in xyz.")

    if chains is None:
        chains = ["A"] * nres
    if len(chains) != nres:
        raise ValueError("chains length must match number of residues in xyz.")

    aa_atom_idx = [
        {name.strip(): i for i, name in enumerate(long) if name is not None}
        for long in aa2long
    ]

    atom_counter = 1
    res_counter = 1
    lines: list[str] = []

    for i in range(nres):
        seq_token = seq_values[i]
        if isinstance(seq_token, str):
            if len(seq_token) == 1:
                res_name = aa_1_to_3[seq_token]
            else:
                res_name = seq_token
        else:
            res_name = num2aa[int(seq_token)]

        chain_id = chains[i] if chains[i] else " "
        atom_map = aa_atom_idx[aa2num[res_name]]

        for atom_name, atom_idx in atom_map.items():
            coord = xyz[i, atom_idx]
            if np.any(np.isnan(coord)):
                raise ValueError(
                    f"Missing coordinate for residue {res_counter} atom {atom_name}."
                )

            element = _infer_element(atom_name, atom_name)
            lines.append(
                f"ATOM  {atom_counter:5d} {atom_name:>4} {res_name:>3} {chain_id}"
                f"{res_counter:4d}    {coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                f"  1.00  0.00          {element:>2}\n"
            )
            atom_counter += 1

        res_counter += 1

    if hetatm is not None:
        dtype_names = set(hetatm.dtype.names or ())
        required = {"atom_name", "res_name", "chain_id", "res_num", "xyz"}
        if not required.issubset(dtype_names):
            missing = sorted(required - dtype_names)
            raise ValueError(f"HETATM array is missing required fields: {missing}")

        for het in hetatm:
            atom_name = str(het["atom_name"]).strip()
            res_name = str(het["res_name"]).strip()
            chain_id = str(het["chain_id"]).strip() or " "
            res_num = int(het["res_num"])
            coord = np.asarray(het["xyz"], dtype=float)
            if coord.shape != (3,):
                raise ValueError("Each HETATM xyz entry must have shape (3,).")
            element = _infer_element(atom_name, atom_name)

            lines.append(
                f"HETATM{atom_counter:5d} {atom_name:>4} {res_name:>3} {chain_id}"
                f"{res_num:4d}    {coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                f"  1.00  0.00          {element:>2}\n"
            )
            atom_counter += 1

    lines.append("END\n")
    outstr = "".join(lines)

    if return_str:
        return outstr

    with open(file_path, "w") as f:
        f.write(outstr)
    return None
