import numpy as np
from typing import Callable, Optional


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
