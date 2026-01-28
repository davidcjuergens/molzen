import numpy as np
from typing import Callable, Optional

from molzen.amino_acids import aa2long, aa2num, oneletter_code
from molzen.ptable import ALL_SYMBOLS


def parse_xyz(xyz_fp):
    """Parse a .xyz with potentially multiple frames in it.

    Args:
        xyz_fp (str): Path to the .xyz file.
    """

    xyzs, elements, comments = [], [], []

    with open(xyz_fp, "r") as f:
        lines = f.readlines()

        i = 0

        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue

            # read number of atoms
            natoms = int(line)
            i += 1

            # read comment line
            comment = lines[i].strip()
            i += 1

            # read atom positions
            xyz = []
            for j in range(natoms):
                if i >= len(lines):
                    break

                split = lines[i].strip().split()
                if len(split) != 4:
                    break  # end of molecule or file

                element = split[0]
                x = float(split[1])
                y = float(split[2])
                z = float(split[3])
                xyz.append([x, y, z])

                # not my favorite way to handle this.
                if len(elements) < natoms:
                    elements.append(element)

                i += 1

            xyzs.append(xyz)
            comments.append(comment)

    return dict(xyz=np.array(xyzs), elements=elements, comments=comments)


def write_xyz(
    filename: str,
    xyz: np.ndarray,
    symbols: list[str],
    comments: Optional[list[str]] = None,
    return_str: bool = False,
):
    """Write XYZ file or return the string representation of the XYZ file.

    Args:
        filename: path  to the output file.
        xyz: coordinates of atoms -- shape (B, N, 3) or (N, 3)
        symbols: list of elements
        comments: list of comments for each frame (optional, default is empty strings)
        return_str: if True, return the string representation instead of writing to a file.
    """
    assert xyz.ndim in (2, 3)

    if xyz.ndim == 2:
        xyz = xyz[None, ...]  # add batch dimension
    B, N, _ = xyz.shape
    assert len(symbols) == N, "Number of symbols must match number of atoms."

    if comments is None:
        comments = ["##"] * B

    outstr = ""

    for b in range(B):
        outstr += f"{N}\n"
        outstr += f"{comments[b]}\n"
        for n in range(N):
            outstr += f"{symbols[n]} {xyz[b, n, 0]} {xyz[b, n, 1]} {xyz[b, n, 2]}\n"

    if return_str:
        return outstr

    with open(filename, "w") as f:
        f.write(outstr)


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


def parse_mol2(mol2_fp: str):
    """Parse a .mol2 file to extract atom information.

    Args:
        mol2_fp (str): Path to the .mol2 file.
    """
    hetatm_data = []

    with open(mol2_fp, "r") as f:
        lines = f.readlines()

        atom_section = False

        for line in lines:
            line = line.strip()
            if line.startswith("@<TRIPOS>ATOM"):
                atom_section = True
                continue
            elif line.startswith("@<TRIPOS>"):
                atom_section = False

            if atom_section:
                split = line.split()
                if len(split) < 6:
                    continue  # malformed line

                atom_idx = int(split[0])
                atom_name = split[1]
                x = float(split[2])
                y = float(split[3])
                z = float(split[4])
                atom_type = split[5]
                element = _infer_element_from_atom_name(atom_name)
                if not element:
                    element = atom_type.split(".")[0].capitalize()
                res_name = split[7] if len(split) > 7 else ""

                het = np.array(
                    (
                        atom_idx,
                        atom_name,
                        atom_type,
                        element,
                        res_name,
                        np.array([x, y, z]),
                    ),
                    dtype=MOL2_HETATM_DTYPES,
                )
                hetatm_data.append(het)

    return dict(hetatm=np.array(hetatm_data))


HETATM_DTYPES = np.dtype(
    [
        ("atom_name", "U4"),
        ("res_name", "U3"),
        ("chain_id", "U1"),
        ("res_num", "i4"),
        ("xyz", "f4", (3,)),
    ]
)


def parse_pdb(pdb_fp: str):
    """Parse a .pdb file to extract atom information. Returns residue-level data for amino acids, atom-level data for HETATM entries.

    Args:
        pdb_fp (str): Path to the .pdb file.
    """
    aa_atom_idx = [
        {name.strip(): i for i, name in enumerate(long) if name is not None}
        for long in aa2long
    ]

    xyz = []
    seq = []
    hetatm_data = []

    with open(pdb_fp, "r") as f:
        lines = f.readlines()

    cur_res_id = None
    cur_xyz = None

    for line in lines:
        if not line.startswith(("ATOM", "HETATM")):
            continue

        if line.startswith("ATOM"):
            atom_name = line[12:16].strip()

            res_name = line[17:20].strip()
            chain_id = line[21]
            resnum = line[22:26].strip()

            res_id = (chain_id, resnum)

            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())

            if res_id != cur_res_id:
                if cur_res_id is not None:
                    # save previous residue
                    xyz.append(cur_xyz)

                # this will raise a KeyError for non-standard residues
                seq.append(oneletter_code[res_name])
                cur_res_id = res_id
                cur_xyz = np.full((27, 3), np.nan)

            aa_index = aa2num[res_name]
            atom_map = aa_atom_idx[aa_index]

            if atom_name in atom_map:
                atom_idx = atom_map[atom_name]
                cur_xyz[atom_idx] = np.array([x, y, z])

        elif line.startswith("HETATM"):
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            chain_id = line[21]
            resnum = line[22:26].strip()
            res_id = (chain_id, resnum)

            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())

            het = np.array(
                (atom_name, res_name, chain_id, int(resnum), np.array([x, y, z])),
                dtype=HETATM_DTYPES,
            )
            hetatm_data.append(het)

    # save last residue
    if cur_res_id is not None:
        xyz.append(cur_xyz)

    return dict(xyz=np.array(xyz), seq="".join(seq), hetatm=np.array(hetatm_data))
