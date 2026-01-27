"""Test pdb parsing"""

from pathlib import Path

from molzen.io.util import parse_pdb


def test_parse_1prw():
    """Test parsing 1PRW pdb file"""
    this_file = Path(__file__)
    pdb_file = this_file.parent / "data" / "1PRW.pdb"

    out = parse_pdb(pdb_file)

    xyz = out["xyz"]  # (Nres, 14, 3)
    seq = out["seq"]  # (Nres, )

    assert xyz.shape == (147, 14, 3)
    assert len(seq) == 147

    assert seq[21] == "D"
    assert xyz[21, 0, 0] == 58.797
    assert seq[54] == "V"
    assert xyz[54, 3, 2] == 0.477

    assert "hetatm" in out
    hetatm = out["hetatm"]  # np.ndarray with dtype HETATM_DTYPES
    assert hetatm["xyz"].shape[1] == 3
