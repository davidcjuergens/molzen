"""Retrieve atomic coordinates from PubChem's PUG REST API."""

import json
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import numpy as np

from molzen.ptable import z_to_symbol
from molzen.bonds import BondGraph


def fetch_pubchem(cid: int | str, *, timeout: float = 30.0) -> dict:
    """Return a Molecule payload containing the first available 3D conformer."""
    if isinstance(cid, bool) or not isinstance(cid, (int, np.integer, str)):
        raise ValueError("cid must be a positive integer or decimal string.")
    if isinstance(cid, str):
        cid = cid.strip()
        if not cid or not cid.isascii() or not cid.isdecimal():
            raise ValueError("cid must be a positive integer or decimal string.")
    cid = int(cid)
    if cid < 1:
        raise ValueError("cid must be a positive integer or decimal string.")
    if not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive and finite.")

    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/JSON"
        "?record_type=3d"
    )
    try:
        with urlopen(url, timeout=timeout) as response:
            raw = response.read()
    except HTTPError as exc:
        if exc.code == 404:
            raise ValueError(
                f"PubChem CID {cid} was not found or has no 3D conformer."
            ) from exc
        raise OSError(
            f"PubChem request for CID {cid} failed: HTTP {exc.code}."
        ) from exc
    except (URLError, TimeoutError) as exc:
        raise OSError(f"PubChem request for CID {cid} failed: {exc}.") from exc

    try:
        compound = json.loads(raw)["PC_Compounds"][0]
        atoms = compound["atoms"]
        atom_ids = atoms["aid"]
        elements = [z_to_symbol[z] for z in atoms["element"]]
        # Coordinate order is defined by its own atom-ID list.
        coords = compound["coords"][0]
        conformer = coords["conformers"][0]
        xyz = np.asarray([conformer[axis] for axis in ("x", "y", "z")], dtype=float).T
        coord_ids = coords["aid"]
        if (
            not atom_ids
            or len(elements) != len(atom_ids)
            or len(set(atom_ids)) != len(atom_ids)
            or len(set(coord_ids)) != len(coord_ids)
            or set(coord_ids) != set(atom_ids)
            or xyz.shape != (len(coord_ids), 3)
            or not np.isfinite(xyz).all()
        ):
            raise ValueError("Inconsistent atoms or coordinates.")
        indices = {aid: i for i, aid in enumerate(coord_ids)}
        xyz = xyz[[indices[aid] for aid in atom_ids]]
        bonds = None
        if "bonds" in compound:
            data = compound["bonds"]
            atom_map = {aid: i for i, aid in enumerate(atom_ids)}
            orders = {
                1: "1",
                2: "2",
                3: "3",
                4: "4",
                5: "dative",
                6: "complex",
                7: "ionic",
                255: "unknown",
            }
            bonds = BondGraph(
                [
                    (atom_map[a], atom_map[b], orders[order], "imported")
                    for a, b, order in zip(
                        data["aid1"], data["aid2"], data["order"], strict=True
                    )
                ],
                status="complete",
            )
    except (KeyError, IndexError, TypeError, ValueError, UnicodeError) as exc:
        raise ValueError(
            f"Invalid PubChem 3D structure response for CID {cid}."
        ) from exc

    return {
        "xyz": xyz[None, ...],
        "elements": elements,
        "comments": [f"PubChem CID {cid} (3D)"],
        "metadata": {"pubchem_cid": cid, "pubchem_url": url},
        "bonds": bonds,
    }
