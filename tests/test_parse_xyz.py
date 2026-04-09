"""Tests for XYZ parser output shape and standardized container."""

import numpy as np
import pytest

from molzen.io.molecule import Molecule


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

    out = Molecule.from_xyz(str(xyz_file))

    assert isinstance(out, Molecule)
    assert out.xyz.shape == (2, 2, 3)
    assert out["elements"] == ["H", "O"]
    assert out["comments"] == ["frame-0", "frame-1"]
    assert "xyz" in out
    assert "seq" not in out
    assert out.get("seq") is None

    with pytest.raises(KeyError):
        _ = out["seq"]

    out_file = tmp_path / "written.xyz"
    out.to_xyz(str(out_file))
    out_reloaded = Molecule.from_xyz(str(out_file))
    assert out_reloaded["comments"] == ["frame-0", "frame-1"]

    xyz_text = out.to_xyz(str(out_file), return_str=True)
    assert "frame-0" in xyz_text


def test_slice_frames_returns_new_molecule_with_selected_frames(tmp_path):
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
        "2\n"
        "frame-2\n"
        "H 0.0 0.2 0.0\n"
        "O 0.0 0.2 1.0\n"
    )

    out = Molecule.from_xyz(str(xyz_file))
    sliced = out.slice_frames(1, 3)
    sliced_from_slice = out.slice_frames(slice(1, 3))

    assert isinstance(sliced, Molecule)
    assert sliced.comments == ["frame-1", "frame-2"]
    assert sliced.atom_records is not None
    assert (
        sliced.atom_records["record_name"].tolist()
        == out.atom_records["record_name"].tolist()
    )
    np.testing.assert_allclose(sliced.xyz, out.xyz[1:3])
    np.testing.assert_allclose(sliced_from_slice.xyz, out.xyz[1:3])
    np.testing.assert_allclose(out[1:3].xyz, out.xyz[1:3])
