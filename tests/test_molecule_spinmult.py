"""Tests for Molecule spin multiplicity metadata."""

from __future__ import annotations

import numpy as np
import pytest

from molzen.io.molecule import Molecule


def test_spinmult_is_mapping_field_and_survives_frame_slice() -> None:
    mol = Molecule(
        xyz=np.array([[0.0, 0.0, 0.0]], dtype=float),
        elements=["O"],
        spinmult="3",
    )

    assert mol.spinmult == 3
    assert mol["spinmult"] == 3
    assert "spinmult" in mol
    assert mol.as_dict()["spinmult"] == 3
    assert mol.as_dict(include_none=True)["spinmult"] == 3
    assert "spinmult=3" in repr(mol)

    multiframe_mol = Molecule(
        xyz=np.array([[[0.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]]], dtype=float),
        elements=["O"],
        comments=["frame-0", "frame-1"],
        spinmult=3,
    )

    sliced = multiframe_mol.slice_frames(slice(1, 2))

    assert sliced.spinmult == 3
    assert sliced["spinmult"] == 3


def test_spinmult_roundtrips_through_npy(tmp_path) -> None:
    mol = Molecule(
        xyz=np.array([[0.0, 0.0, 0.0]], dtype=float),
        elements=["N"],
        spinmult=2,
    )

    npy_file = tmp_path / "mol.npy"
    mol.to_npy(str(npy_file))

    parsed = Molecule.from_npy(str(npy_file))

    assert parsed.spinmult == 2
    assert parsed["spinmult"] == 2


def test_spinmult_roundtrips_through_hdf5(tmp_path) -> None:
    pytest.importorskip("h5py")

    mol = Molecule(
        xyz=np.array([[0.0, 0.0, 0.0]], dtype=float),
        elements=["N"],
        spinmult=4,
    )

    h5_file = tmp_path / "mol.hdf5"
    mol.to_hdf5(str(h5_file))

    parsed = Molecule.from_hdf5(str(h5_file))

    assert parsed.spinmult == 4
    assert parsed["spinmult"] == 4


@pytest.mark.parametrize("spinmult", [0, -1, "1.5", True, 1.5])
def test_spinmult_rejects_non_positive_integers(spinmult) -> None:
    with pytest.raises(ValueError, match="spinmult must be a positive integer"):
        Molecule(
            xyz=np.array([[0.0, 0.0, 0.0]], dtype=float),
            elements=["H"],
            spinmult=spinmult,
        )
