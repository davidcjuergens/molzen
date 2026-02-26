"""XYZ format readers/writers that operate on plain data."""

from __future__ import annotations

from typing import Any

import numpy as np


def parse_xyz(xyz_fp: str) -> dict[str, Any]:
    """Parse a `.xyz` file with one or more frames."""
    xyzs: list[list[list[float]]] = []
    comments: list[str] = []
    elements: list[str] | None = None

    with open(xyz_fp, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        try:
            natoms = int(line)
        except ValueError as exc:
            raise ValueError(f"Invalid atom count at line {i + 1}: {line!r}") from exc
        i += 1

        if i >= len(lines):
            raise ValueError("Unexpected end-of-file while reading XYZ comment line.")
        comment = lines[i].rstrip("\n")
        comments.append(comment)
        i += 1

        frame_xyz: list[list[float]] = []
        frame_elements: list[str] = []
        for atom_i in range(natoms):
            if i >= len(lines):
                raise ValueError(
                    f"Unexpected end-of-file while reading atom {atom_i + 1} "
                    f"of frame {len(xyzs) + 1}."
                )

            split = lines[i].split()
            if len(split) != 4:
                raise ValueError(
                    f"Malformed XYZ atom line {i + 1}: expected 4 columns, got "
                    f"{len(split)}."
                )

            element = split[0]
            try:
                x = float(split[1])
                y = float(split[2])
                z = float(split[3])
            except ValueError as exc:
                raise ValueError(f"Invalid XYZ coordinates at line {i + 1}.") from exc

            frame_xyz.append([x, y, z])
            frame_elements.append(element)
            i += 1

        if elements is None:
            elements = frame_elements
        elif frame_elements != elements:
            raise ValueError("Element order changed between XYZ frames.")

        xyzs.append(frame_xyz)

    if not xyzs:
        raise ValueError(f"No coordinate frames found in XYZ file: {xyz_fp}")

    return {
        "xyz": np.array(xyzs, dtype=float),
        "elements": elements if elements is not None else [],
        "comments": comments,
    }


def write_xyz(
    filename: str,
    xyz: np.ndarray,
    symbols: list[str],
    comments: list[str] | None = None,
    return_str: bool = False,
) -> str | None:
    """Write XYZ text to disk or return it as a string."""
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim not in (2, 3):
        raise ValueError("xyz must have shape (N, 3) or (B, N, 3).")
    if xyz.ndim == 2:
        xyz = xyz[None, ...]

    batch, natoms, ncoords = xyz.shape
    if ncoords != 3:
        raise ValueError("xyz must have a final coordinate dimension of size 3.")
    if len(symbols) != natoms:
        raise ValueError("Number of symbols must match number of atoms.")

    if comments is None:
        comments = ["##"] * batch
    if len(comments) != batch:
        raise ValueError("Number of comments must match number of frames.")

    chunks: list[str] = []
    for b in range(batch):
        chunks.append(f"{natoms}\n")
        chunks.append(f"{comments[b]}\n")
        for n in range(natoms):
            chunks.append(
                f"{symbols[n]} {xyz[b, n, 0]} {xyz[b, n, 1]} {xyz[b, n, 2]}\n"
            )

    outstr = "".join(chunks)
    if return_str:
        return outstr

    with open(filename, "w") as f:
        f.write(outstr)
    return None
