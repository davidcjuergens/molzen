"""Atom-axis joins and rigid motions of Molecule."""

import numpy as np
import pytest

from molzen.bonds import BondGraph
from molzen.io.molecule import Molecule
from molzen.kinematics import apply_rigid_motion, best_fit_frame, plane_alignment


def _pair(xyz, elements, **kwargs):
    return Molecule(xyz=np.array(xyz, dtype=float), elements=elements, **kwargs)


def test_join_remaps_colliding_atom_indices_and_bonds():
    left = _pair(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], ["C", "O"], metadata={"name": "left"}
    )
    right = _pair(
        [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], ["N", "H"], metadata={"name": "right"}
    )
    left.bonds = BondGraph([(0, 1, "1", "manual")], status="complete")
    right.bonds = BondGraph([(0, 1, "2", "imported")], status="complete")
    left_xyz = np.array(left.xyz)
    right_ids = right.atom_records["atom_index"].copy()

    joined = Molecule.join([left, right])

    assert len(joined.atom_records) == 4
    assert joined.atom_records["atom_index"].tolist() == [0, 1, 2, 3]
    assert joined.atom_records["serial"].tolist() == [1, 2, 3, 4]
    assert len(joined.bonds) == 2
    assert joined.bonds.status == "complete"
    edges = {
        (int(edge["a"]), int(edge["b"]), str(edge["order"]), str(edge["source"]))
        for edge in joined.bonds.records
    }
    assert edges == {(0, 1, "1", "manual"), (2, 3, "2", "imported")}
    np.testing.assert_allclose(left.xyz, left_xyz)
    assert left.bonds.status == "complete"
    assert len(left.bonds) == 1
    np.testing.assert_array_equal(right.atom_records["atom_index"], right_ids)
    assert joined.metadata["join"]["segments"][0]["atom_index_start"] == 0
    assert joined.metadata["join"]["segments"][0]["atom_index_stop"] == 2
    assert joined.metadata["join"]["segments"][1]["metadata"] == {"name": "right"}
    assert "name" not in joined.metadata


def test_join_bond_status_rules():
    def bonded(status):
        mol = _pair([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], ["C", "C"])
        mol.bonds = BondGraph([(0, 1, "1", "manual")], status=status)
        return mol

    complete = bonded("complete")
    partial = bonded("partial")
    unknown = _pair([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], ["C", "C"])

    both_complete = Molecule.join([complete, bonded("complete")])
    assert both_complete.bonds.status == "complete"
    assert len(both_complete.bonds) == 2

    mixed = Molecule.join([complete, partial])
    assert mixed.bonds.status == "partial"
    assert len(mixed.bonds) == 2

    missing = Molecule.join([complete, unknown])
    assert missing.bonds.status == "partial"
    assert len(missing.bonds) == 1

    absent = Molecule.join([unknown, _pair([[2.0, 0.0, 0.0]], ["O"])])
    assert absent.bonds is None


def test_join_keeps_colliding_residues_distinct_and_records_the_remap():
    plain = _pair(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ["C", "O"],
        metadata={"source": {"id": "pubchem"}},
    )
    other = _pair([[2.0, 0.0, 0.0]], ["N"], metadata={"source": {"id": "other"}})
    chain_a = _pair([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], ["C", "C"])
    chain_b = _pair([[3.0, 0.0, 0.0]], ["O"])
    chain_a.atom_records["chain_id"] = ["A", "A"]
    chain_a.atom_records["res_name"] = ["ALA", "ALA"]
    chain_b.atom_records["chain_id"] = ["B"]
    chain_b.atom_records["res_name"] = ["HOH"]

    joined = Molecule.join([plain, other])
    distinct = Molecule.join([chain_a, chain_b])

    assert joined.atom_records["res_num"].tolist() == [1, 1, 2]
    assert len(set(joined.atom_records["residue_index"].tolist())) == 2
    assert joined.metadata["join"]["segments"][0]["residue_remaps"] == []
    assert joined.metadata["join"]["segments"][1]["residue_remaps"] == [
        {
            "original": {
                "record_name": "HETATM",
                "res_num": 1,
                "chain_id": "",
                "i_code": "",
                "res_name": "MOL",
            },
            "remapped": {
                "record_name": "HETATM",
                "res_num": 2,
                "chain_id": "",
                "i_code": "",
                "res_name": "MOL",
            },
        }
    ]
    joined.metadata["join"]["segments"][0]["metadata"]["source"]["id"] = "changed"
    assert plain.metadata["source"]["id"] == "pubchem"
    assert distinct.atom_records["chain_id"].tolist() == ["A", "A", "B"]
    assert distinct.atom_records["res_num"].tolist() == [1, 1, 1]
    assert distinct.metadata["join"]["segments"][1]["residue_remaps"] == []


def test_join_rejects_empty_input_and_dangling_bonds():
    with pytest.raises(ValueError, match="At least one molecule is required"):
        Molecule.join([])

    mol = _pair([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], ["C", "C"])
    mol._bonds = BondGraph([(0, 5, "1", "manual")], status="complete")
    with pytest.raises(ValueError, match="Bond endpoint does not exist"):
        Molecule.join([mol])


def test_join_rejects_frame_count_mismatch():
    one = _pair([[0.0, 0.0, 0.0]], ["C"])
    two = _pair(
        [[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]],
        ["C"],
    )

    with pytest.raises(ValueError, match="different frame counts"):
        Molecule.join([one, two])


def test_join_comments_and_spinmult_rules():
    left = _pair([[0.0, 0.0, 0.0]], ["C"], comments=["same"], spinmult=1)
    right = _pair([[1.0, 0.0, 0.0]], ["O"], comments=["same"], spinmult=None)
    different = _pair([[1.0, 0.0, 0.0]], ["O"], comments=["other"], spinmult=1)
    triplet = _pair([[1.0, 0.0, 0.0]], ["O"], comments=["same"], spinmult=3)

    assert Molecule.join([left, right]).comments == ["same"]
    assert Molecule.join([left, right]).spinmult == 1
    with pytest.raises(ValueError, match="different comments"):
        Molecule.join([left, different])
    joined = Molecule.join([left, different], comments=["explicit"])
    assert joined.comments == ["explicit"]
    with pytest.raises(ValueError, match="different spinmults"):
        Molecule.join([left, triplet])


def test_join_keeps_atom_order_in_every_frame():
    left = _pair(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
        ],
        ["C", "O"],
        comments=["f0", "f1"],
    )
    right = _pair(
        [
            [[2.0, 0.0, 0.0]],
            [[2.0, 0.0, 1.0]],
        ],
        ["N"],
        comments=["f0", "f1"],
    )

    joined = Molecule.join([left, right])

    assert joined.xyz.shape == (2, 3, 3)
    np.testing.assert_allclose(joined.xyz[0, :, 0], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(joined.xyz[1, :, 0], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(joined.xyz[:, :, 2], [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])


def test_join_rejects_excited_state_records_unless_dropped():
    left = _pair(
        [[0.0, 0.0, 0.0]],
        ["C"],
        excited_state_records=[
            {"frame_index": 0, "state_j": 0, "total_energy_au": -1.0}
        ],
    )
    right = _pair([[1.0, 0.0, 0.0]], ["O"])

    with pytest.raises(ValueError, match="excited_state_records"):
        Molecule.join([left, right])
    joined = Molecule.join([left, right], drop_excited_state_records=True)
    assert joined.excited_state_records is None
    assert left.excited_state_records[0]["total_energy_au"] == -1.0


def test_join_one_molecule_reindexes_without_changing_the_input():
    mol = _pair([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], ["C", "H"])
    mol.atom_records["atom_index"] = [4, 9]
    mol.bonds = BondGraph([(4, 9, "1", "template")], status="partial")

    joined = Molecule.join([mol])

    assert joined.atom_records["atom_index"].tolist() == [0, 1]
    assert int(joined.bonds.records[0]["a"]) == 0
    assert int(joined.bonds.records[0]["b"]) == 1
    assert joined.bonds.status == "partial"
    assert mol.atom_records["atom_index"].tolist() == [4, 9]
    assert int(mol.bonds.records[0]["a"]) == 4


def test_translated_rotated_and_aligned_leave_the_input_unchanged():
    frame0 = np.array(
        [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.5, 2.0, 0.5]],
        dtype=float,
    )
    frame1 = frame0 + np.array([0.4, -0.2, 0.7])
    mol = Molecule(
        xyz=np.stack([frame0, frame1]),
        elements=["C", "C", "C", "H"],
        comments=["a", "b"],
        spinmult=1,
        metadata={"job": "geom"},
        excited_state_records=[
            {"frame_index": 1, "state_j": 0, "total_energy_au": -2.0}
        ],
    )
    mol.bonds = BondGraph(
        [(0, 1, "1", "manual"), (1, 2, "2", "imported")], status="complete"
    )
    original = np.array(mol.xyz)

    moved = mol.translated([1.0, 2.0, 3.0])
    turned = mol.rotated([0.0, 0.0, 1.0], 90.0, origin=[0.0, 0.0, 0.0])
    flat = mol.aligned_to_plane(normal=(0.0, 0.0, 1.0))

    np.testing.assert_allclose(mol.xyz, original)
    assert mol.bonds.status == "complete"
    assert len(mol.bonds) == 2
    assert mol.comments == ["a", "b"]
    assert mol.spinmult == 1
    assert mol.metadata == {"job": "geom"}
    assert mol.excited_state_records[0]["frame_index"] == 1
    for result in (moved, turned, flat):
        assert result is not mol
        assert result.bonds.status == "complete"
        assert len(result.bonds) == 2
        assert result.atom_records["atom_index"].tolist() == [0, 1, 2, 3]
        assert result.comments == ["a", "b"]
        assert result.spinmult == 1
        assert result.metadata == {"job": "geom"}
        assert result.excited_state_records == mol.excited_state_records
    np.testing.assert_allclose(
        moved.xyz, original + np.array([1.0, 2.0, 3.0]), atol=1e-5
    )


def test_rotation_is_right_handed_about_frame0_centroid():
    mol = Molecule(
        xyz=np.array(
            [
                [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
                [[1.0, 2.0, 0.0], [-1.0, 2.0, 0.0]],
            ],
            dtype=float,
        ),
        elements=["C", "C"],
    )

    turned = mol.rotated([0.0, 0.0, 2.0], 90.0)

    np.testing.assert_allclose(
        turned.xyz[0], [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], atol=1e-5
    )
    np.testing.assert_allclose(
        turned.xyz[1], [[-2.0, 1.0, 0.0], [-2.0, -1.0, 0.0]], atol=1e-5
    )
    with pytest.raises(ValueError, match="nonzero length"):
        mol.rotated([0.0, 0.0, 0.0], 90.0)


def test_aligned_to_plane_uses_one_frame_for_every_frame():
    frame0 = np.array(
        [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
        dtype=float,
    )
    offset = np.array([0.4, -0.2, 0.7])
    frame1 = frame0 + offset
    mol = Molecule(xyz=np.stack([frame0, frame1]), elements=["C", "N", "O"])

    flat = mol.aligned_to_plane()

    _, source = best_fit_frame(frame0)
    origin, rotation, translation = plane_alignment(frame0, (0.0, 0.0, 1.0))
    np.testing.assert_allclose(rotation @ source[:, 2], [0.0, 0.0, 1.0], atol=1e-6)
    expected = apply_rigid_motion(
        np.stack([frame0, frame1]), rotation, origin=origin, translation=translation
    )
    np.testing.assert_allclose(flat.xyz[0].mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(flat.xyz[0, :, 2], 0.0, atol=1e-5)
    np.testing.assert_allclose(flat.xyz, expected, atol=1e-5)
    assert not np.allclose(flat.xyz[1].mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(mol.xyz[0], frame0)


def test_aligned_to_plane_requires_three_atoms_and_skips_hydrogens_by_default():
    heavy = Molecule(
        xyz=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float),
        elements=["C", "C", "H"],
    )
    with pytest.raises(ValueError, match="At least three atoms"):
        heavy.aligned_to_plane()
    with pytest.raises(ValueError, match="At least three atoms"):
        Molecule(
            xyz=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
            elements=["C", "O"],
        ).aligned_to_plane()
