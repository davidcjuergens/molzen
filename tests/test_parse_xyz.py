"""Tests for XYZ parser output shape and standardized container."""

import pytest

from molzen.io.molecule import Molecule
from molzen.io.util import parse_xyz


def test_parse_xyz_returns_molecule_and_dict_compat(tmp_path):
    xyz_file = tmp_path / "frames.xyz"
    xyz_file.write_text(
        "2\n"
        "frame-0\n"
        "H 0.0 0.0 0.0\n"
        "O 0.0 0.0 1.0\n"
        "2\n"
        "frame-1\n"
        "H 0.0 0.1 0.0\n"
        "O 0.0 0.1 1.0\n"
    )

    out = parse_xyz(str(xyz_file))

    assert isinstance(out, Molecule)
    assert out.xyz.shape == (2, 2, 3)
    assert out["elements"] == ["H", "O"]
    assert out["comments"] == ["frame-0", "frame-1"]
    assert "xyz" in out
    assert "seq" not in out
    assert out.get("seq") is None

    with pytest.raises(KeyError):
        _ = out["seq"]
