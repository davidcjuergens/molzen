"""Tests for Molecule class format methods."""

import numpy as np
import pytest

from molzen.io.molecule import Molecule


def test_mol2_roundtrip(tmp_path):
    mol = Molecule(
        xyz=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float),
        atom_names=["O1", "H1"],
        elements=["O", "H"],
    )

    mol2_file = tmp_path / "test.mol2"
    mol.to_mol2(str(mol2_file))
    parsed = Molecule.from_mol2(str(mol2_file))

    assert parsed.xyz.shape == (2, 3)
    assert parsed["atom_names"] == ["O1", "H1"]
    assert parsed["elements"] == ["O", "H"]
    assert parsed.atom_records is not None
    assert parsed.atom_records["record_name"].tolist() == ["HETATM", "HETATM"]

    mol2_text = mol.to_mol2(str(mol2_file), return_str=True)
    assert "@<TRIPOS>ATOM" in mol2_text


def test_xyz_to_pdb_uses_hetatm_without_seq(tmp_path):
    xyz_file = tmp_path / "single.xyz"
    xyz_file.write_text("2\nframe-0\nC 0.0 0.0 0.0\nO 0.0 0.0 1.2\n")

    mol = Molecule.from_xyz(str(xyz_file))
    assert mol.atom_records is not None
    assert mol.atom_records["record_name"].tolist() == ["HETATM", "HETATM"]

    mol.seq = ""
    pdb_file = tmp_path / "single.pdb"
    pdb_text = mol.to_pdb(str(pdb_file), return_str=True)
    assert "HETATM" in pdb_text
    assert "ATOM  " not in pdb_text

    mol.to_pdb(str(pdb_file))
    parsed = Molecule.from_pdb(str(pdb_file))
    assert parsed.xyz.shape == (2, 3)
    assert parsed.polymer_xyz is None
    assert parsed.atom_records["record_name"].tolist() == ["HETATM", "HETATM"]
    assert parsed.atom_records["element"].tolist() == ["C", "O"]


def test_npy_roundtrip(tmp_path):
    mol = Molecule(
        xyz=np.array([[1.0, 2.0, 3.0]], dtype=float),
        elements=["C"],
        comments=["single-frame"],
        metadata={"source": "unit-test"},
    )

    npy_file = tmp_path / "mol.npy"
    mol.to_npy(str(npy_file))
    parsed = Molecule.from_npy(str(npy_file))

    np.testing.assert_allclose(parsed["xyz"], mol["xyz"])
    assert parsed["elements"] == ["C"]
    assert parsed.metadata["source"] == "unit-test"
    assert parsed.atom_records is not None
    assert parsed.atom_records["element"].tolist() == ["C"]

    blob = mol.to_npy(str(npy_file), return_bytes=True)
    assert isinstance(blob, bytes)
    assert len(blob) > 0


def test_hdf5_roundtrip(tmp_path):
    pytest.importorskip("h5py")

    mol = Molecule(
        xyz=np.array([[0.1, 0.2, 0.3]], dtype=float),
        elements=["He"],
        metadata={"kind": "hdf5"},
    )

    h5_file = tmp_path / "mol.hdf5"
    mol.to_hdf5(str(h5_file))
    parsed = Molecule.from_hdf5(str(h5_file))

    np.testing.assert_allclose(parsed["xyz"], mol["xyz"])
    assert parsed["elements"] == ["He"]
    assert parsed.metadata["kind"] == "hdf5"
    assert parsed.atom_records is not None
    assert parsed.atom_records["element"].tolist() == ["He"]
