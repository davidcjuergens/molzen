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
        excited_state_records=[
            {
                "frame_index": 0,
                "section_idx": 0,
                "state_i": 0,
                "state_j": 1,
                "multiplicity": "singlet",
                "total_energy_au": -1.0,
            }
        ],
    )

    npy_file = tmp_path / "mol.npy"
    mol.to_npy(str(npy_file))
    parsed = Molecule.from_npy(str(npy_file))

    np.testing.assert_allclose(parsed["xyz"], mol["xyz"])
    assert parsed["elements"] == ["C"]
    assert parsed.metadata["source"] == "unit-test"
    assert parsed.excited_state_records == mol.excited_state_records
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
        excited_state_records=[
            {
                "frame_index": 0,
                "section_idx": 0,
                "state_i": 0,
                "state_j": 0,
                "multiplicity": "singlet",
                "total_energy_au": -2.0,
            }
        ],
    )

    h5_file = tmp_path / "mol.hdf5"
    mol.to_hdf5(str(h5_file))
    parsed = Molecule.from_hdf5(str(h5_file))

    np.testing.assert_allclose(parsed["xyz"], mol["xyz"])
    assert parsed["elements"] == ["He"]
    assert parsed.metadata["kind"] == "hdf5"
    assert parsed.excited_state_records == mol.excited_state_records
    assert parsed.atom_records is not None
    assert parsed.atom_records["element"].tolist() == ["He"]


def test_dmap_returns_pairwise_distances_for_each_frame():
    mol = Molecule(
        xyz=np.array(
            [
                [
                    [0.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [0.0, 4.0, 0.0],
                ],
                [
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 6.0],
                    [1.0, 5.0, 1.0],
                ],
            ],
            dtype=float,
        ),
        elements=["C", "H", "O"],
        comments=["frame-0", "frame-1"],
    )

    expected = np.array(
        [
            [
                [0.0, 3.0, 4.0],
                [3.0, 0.0, 5.0],
                [4.0, 5.0, 0.0],
            ],
            [
                [0.0, 5.0, 4.0],
                [5.0, 0.0, np.sqrt(41.0)],
                [4.0, np.sqrt(41.0), 0.0],
            ],
        ],
        dtype=float,
    )

    assert mol.dmap.shape == (2, 3, 3)
    np.testing.assert_allclose(mol.dmap, expected)
    np.testing.assert_allclose(mol.dmap, np.swapaxes(mol.dmap, 1, 2))
    np.testing.assert_allclose(np.diagonal(mol.dmap, axis1=1, axis2=2), 0.0)


def test_shape_returns_xyz_shape():
    mol = Molecule(
        xyz=np.array(
            [
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0],
                ],
            ],
            dtype=float,
        ),
        elements=["H", "O"],
        comments=["frame-0", "frame-1"],
    )

    assert mol.shape == mol.xyz.shape == (2, 2, 3)


def test_dmap_handles_single_frame_xyz_without_public_frame_axis():
    mol = Molecule(
        xyz=np.array(
            [
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, 4.0, 0.0],
            ],
            dtype=float,
        ),
        elements=["C", "H", "O"],
    )

    assert mol.xyz.shape == (3, 3)
    assert mol._atom_records["coords"].shape == (3, 1, 3)
    np.testing.assert_allclose(
        mol.dmap,
        np.array(
            [
                [
                    [0.0, 3.0, 4.0],
                    [3.0, 0.0, 5.0],
                    [4.0, 5.0, 0.0],
                ]
            ],
            dtype=float,
        ),
    )


def test_pop_removes_and_returns_atom_record():
    mol = Molecule(
        xyz=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
        elements=["C", "H", "O"],
    )
    mol.metadata = {"pdb_raw_lines": ["ATOM"], "source": "unit-test"}

    removed = mol.pop(1)

    assert removed["element"] == "H"
    np.testing.assert_allclose(removed["coords"], [[1.0, 0.0, 0.0]])
    assert mol.atom_records is not None
    assert mol.atom_records["element"].tolist() == ["C", "O"]
    assert mol.atom_records["coords"].shape == (2, 1, 3)
    assert "pdb_raw_lines" not in mol.metadata
    assert mol.metadata["source"] == "unit-test"


def test_slice_frames_remaps_excited_state_record_frame_indices():
    mol = Molecule(
        xyz=np.array(
            [
                [[0.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0]],
                [[2.0, 0.0, 0.0]],
            ],
            dtype=float,
        ),
        elements=["H"],
        excited_state_records=[
            {"frame_index": 0, "state_j": 0, "total_energy_au": -1.0},
            {"frame_index": 1, "state_j": 0, "total_energy_au": -2.0},
            {"frame_index": 2, "state_j": 0, "total_energy_au": -3.0},
        ],
    )

    sliced = mol.slice_frames(1, 3)

    assert sliced.excited_state_records == [
        {"frame_index": 0, "state_j": 0, "total_energy_au": -2.0},
        {"frame_index": 1, "state_j": 0, "total_energy_au": -3.0},
    ]


def test_cat_frames_concatenates_coordinates_comments_and_excited_states():
    mol_a = Molecule(
        xyz=np.array(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0], [1.0, 0.0, 1.0]],
            ],
            dtype=float,
        ),
        elements=["C", "O"],
        comments=["a0", "a1"],
        spinmult=1,
        metadata={"job": "a"},
        excited_state_records=[
            {"frame_index": 0, "state_j": 0, "total_energy_au": -1.0},
            {"frame_index": 1, "state_j": 0, "total_energy_au": -2.0},
        ],
    )
    mol_b = Molecule(
        xyz=np.array([[2.0, 0.0, 0.0], [2.0, 0.0, 1.0]], dtype=float),
        elements=["C", "O"],
        comments=["b0"],
        spinmult=1,
        metadata={"job": "b"},
        excited_state_records=[
            {"frame_index": 0, "state_j": 0, "total_energy_au": -3.0},
            {"state_j": 1, "total_energy_au": -2.5},
        ],
    )
    mol_b.comments = None

    combined = Molecule.cat_frames([mol_a, mol_b])

    assert combined.xyz.shape == (3, 2, 3)
    np.testing.assert_allclose(
        combined.xyz,
        np.array(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0], [1.0, 0.0, 1.0]],
                [[2.0, 0.0, 0.0], [2.0, 0.0, 1.0]],
            ],
            dtype=float,
        ),
    )
    assert combined.comments == ["a0", "a1", ""]
    assert combined.spinmult == 1
    assert combined.excited_state_records == [
        {"frame_index": 0, "state_j": 0, "total_energy_au": -1.0},
        {"frame_index": 1, "state_j": 0, "total_energy_au": -2.0},
        {"frame_index": 2, "state_j": 0, "total_energy_au": -3.0},
    ]
    assert combined.metadata["cat_frames"]["segments"] == [
        {
            "molecule_index": 0,
            "frame_start": 0,
            "frame_stop": 2,
            "metadata": {"job": "a"},
        },
        {
            "molecule_index": 1,
            "frame_start": 2,
            "frame_stop": 3,
            "metadata": {"job": "b"},
        },
    ]
    assert combined.metadata["cat_frames"]["frame_boundaries"] == [2]


def test_cat_frames_accepts_three_or_more_molecules():
    mols = [
        Molecule(
            xyz=np.array([[float(i), 0.0, 0.0]], dtype=float),
            elements=["H"],
        )
        for i in range(3)
    ]

    combined = Molecule.cat_frames(mols)

    assert combined.xyz.shape == (3, 1, 3)
    np.testing.assert_allclose(
        combined.xyz[:, 0, 0],
        np.array([0.0, 1.0, 2.0], dtype=float),
    )


def test_slice_frames_remaps_cat_frame_boundaries():
    mols = [
        Molecule(
            xyz=np.array(
                [
                    [[float(i), 0.0, 0.0]],
                    [[float(i) + 0.5, 0.0, 0.0]],
                ],
                dtype=float,
            ),
            elements=["H"],
        )
        for i in (0, 1)
    ]
    combined = Molecule.cat_frames(mols)

    sliced = combined.slice_frames(1, 4)

    assert combined.metadata["cat_frames"]["frame_boundaries"] == [2]
    assert sliced.metadata["cat_frames"]["frame_boundaries"] == [1]


def test_cat_frames_rejects_mismatched_atom_metadata():
    mol_a = Molecule(
        xyz=np.array([[0.0, 0.0, 0.0]], dtype=float),
        elements=["C"],
    )
    mol_b = Molecule(
        xyz=np.array([[1.0, 0.0, 0.0]], dtype=float),
        elements=["O"],
    )

    with pytest.raises(ValueError, match="field 'element' differs"):
        Molecule.cat_frames([mol_a, mol_b])


def test_cat_frames_rejects_conflicting_spinmults():
    mol_a = Molecule(
        xyz=np.array([[0.0, 0.0, 0.0]], dtype=float),
        elements=["H"],
        spinmult=1,
    )
    mol_b = Molecule(
        xyz=np.array([[1.0, 0.0, 0.0]], dtype=float),
        elements=["H"],
        spinmult=3,
    )

    with pytest.raises(ValueError, match="different spinmults"):
        Molecule.cat_frames([mol_a, mol_b])


def test_terachem_records_with_frame_indices_maps_section_idx():
    records = [
        {"section_idx": 0, "state_j": 0, "total_energy_au": -1.0},
        {"section_idx": 2, "state_j": 1, "total_energy_au": -0.5},
    ]

    assert Molecule._terachem_records_with_frame_indices(records) == [
        {
            "section_idx": 0,
            "frame_index": 0,
            "state_j": 0,
            "total_energy_au": -1.0,
        },
        {
            "section_idx": 2,
            "frame_index": 2,
            "state_j": 1,
            "total_energy_au": -0.5,
        },
    ]


def test_excited_state_record_frame_indices_must_match_frame_count():
    with pytest.raises(ValueError, match="frame_index out of range"):
        Molecule(
            xyz=np.array(
                [
                    [[0.0, 0.0, 0.0]],
                    [[1.0, 0.0, 0.0]],
                ],
                dtype=float,
            ),
            elements=["H"],
            excited_state_records=[
                {"frame_index": 2, "state_j": 0, "total_energy_au": -1.0}
            ],
        )


def test_excited_state_record_assignment_validates_frame_indices():
    mol = Molecule(
        xyz=np.array(
            [
                [[0.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0]],
            ],
            dtype=float,
        ),
        elements=["H"],
    )

    with pytest.raises(ValueError, match="frame_index out of range"):
        mol.excited_state_records = [
            {"frame_index": 5, "state_j": 0, "total_energy_au": -1.0}
        ]
