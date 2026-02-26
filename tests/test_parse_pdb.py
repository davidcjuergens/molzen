"""Test pdb parsing"""

from pathlib import Path

from molzen.io.molecule import Molecule


def test_parse_1prw(tmp_path):
    """Test parsing 1PRW pdb file"""
    this_file = Path(__file__)
    pdb_file = this_file.parent / "data" / "1PRW.pdb"

    out = Molecule.from_pdb(str(pdb_file))
    assert isinstance(out, Molecule)

    xyz = out["xyz"]  # (Nres, 27, 3)
    seq = out["seq"]  # (Nres, )

    assert xyz.shape == (147, 27, 3)
    assert len(seq) == 147

    assert seq[21] == "D"
    assert xyz[21, 0, 0] == 58.797
    assert seq[54] == "V"
    assert xyz[54, 3, 2] == 0.477

    assert "hetatm" in out
    hetatm = out["hetatm"]  # np.ndarray with dtype HETATM_DTYPES
    assert hetatm["xyz"].shape[1] == 3

    out_file = tmp_path / "roundtrip.pdb"
    out.to_pdb(str(out_file))
    out_roundtrip = Molecule.from_pdb(str(out_file))
    assert out_roundtrip["xyz"].shape == xyz.shape
    assert out_roundtrip["seq"] == seq
    assert "pdb_records" in out_roundtrip.metadata
