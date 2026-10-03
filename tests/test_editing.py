"""Connectivity, transactional edits, and serialization of edited structures."""

import io
import json

import numpy as np
import pytest

from molzen.bonds import BondGraph
from molzen.editing import EditSession
from molzen.io import Molecule
from molzen.io import pubchem


@pytest.fixture
def methane():
    directions = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(
        3
    )
    mol = Molecule(
        elements=["C", "H", "H", "H", "H"],
        xyz=np.vstack([np.zeros(3), 1.09 * directions]),
        metadata={"energy": -1},
        spinmult=1,
    )
    mol.atom_records["atom_index"] = [10, 20, 30, 40, 50]
    mol.bonds = BondGraph(
        [(10, i, "1", "imported") for i in (20, 30, 40, 50)], status="complete"
    )
    return mol


def test_methyl_geometry_and_undo(methane):
    session = EditSession(methane)
    session.apply("replace_hydrogen", atom_id=20, torsion=45)
    result = session.molecule
    assert len(result.atom_records) == 8
    assert len(result.bonds) == 7
    assert len(methane.atom_records) == 5
    assert methane.elements[1] == "H"
    assert result.spinmult is None
    assert "energy" not in result.metadata
    assert result.metadata["source_metadata"]["energy"] == -1
    xyz = result.atom_records["coords"][:, 0]
    assert np.linalg.norm(xyz[1] - xyz[0]) == pytest.approx(1.52)
    vectors = xyz[-3:] - xyz[1]
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.09, atol=1e-6)
    unit = vectors / np.linalg.norm(vectors, axis=1)[:, None]
    np.testing.assert_allclose(
        (unit @ unit.T)[np.triu_indices(3, 1)], -1 / 3, atol=1e-6
    )
    result.atom_records["element"][0] = "O"
    assert session.molecule.elements[0] == "C"  # result is an independent copy
    session.undo()
    assert session.molecule.bonds.to_dict() == methane.bonds.to_dict()
    assert session.molecule.spinmult == 1
    session.redo()
    assert len(session.molecule.atom_records) == 8


def test_failed_edit_is_atomic(methane):
    session = EditSession(methane)
    with pytest.raises(ValueError, match="hydrogen"):
        session.apply("replace_hydrogen", atom_id=10)
    assert not session.can_undo
    assert session.revision == 0
    session.apply("set_element", atom_id=20, element="F")
    assert session.molecule.elements[1] == "F"
    session.undo()
    session.apply("delete_atom", atom_id=30)
    assert session.molecule.atom_records["atom_index"].tolist() == [10, 20, 40, 50]
    assert session.molecule.bonds.neighbors(10) == [20, 40, 50]
    assert not session.can_redo


def test_bond_validation(methane):
    for rows in [
        [(1, 1, "1", "manual")],
        [(1, 2, "1", "manual"), (2, 1, "2", "manual")],
        [(1.5, 2, "1", "manual")],
        [(1, 2, "nonsense", "manual")],
    ]:
        with pytest.raises(ValueError):
            BondGraph(rows)
    with pytest.raises(ValueError, match="endpoint"):
        methane.bonds = BondGraph([(10, 999, "1", "manual")])
    assert len(methane.bonds) == 4
    copied = methane.bonds.records
    copied["a"] = 999
    assert methane.bonds.neighbors(10) == [20, 30, 40, 50]


def test_inference_requires_acceptance_and_confirmation(methane):
    methane.bonds = None
    session = EditSession(methane)
    preview = session.suggest_bonds()
    assert len(preview.bonds) == 4
    assert session.molecule.bonds is None
    assert set(preview.bonds.records["order"]) == {"unknown"}
    session.apply("accept_bonds")
    assert session.molecule.bonds.status == "partial"
    with pytest.raises(ValueError, match="Confirm"):
        session.apply("replace_hydrogen", atom_id=20)
    session.apply("set_bond", a=10, b=20, order="1")
    session.apply("replace_hydrogen", atom_id=20)
    assert len(session.molecule.atom_records) == 8


def test_inference_skips_metals_alternates_and_preserves_edges():
    mol = Molecule(
        elements=["C", "H", "H", "Fe"],
        xyz=np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 0, 1.5]]),
    )
    mol.atom_records["alt_loc"] = ["A", "B", "A", ""]
    mol.bonds = BondGraph([(0, 2, "1", "manual")])
    result = mol.infer_bonds()
    assert len(result.bonds) == 0
    assert any("Fe" in diagnostic for diagnostic in result.diagnostics)
    assert mol.bonds.records["source"].tolist() == ["manual"]


@pytest.mark.parametrize("suffix", ["npy", "hdf5", "mol2", "pdb"])
def test_edited_roundtrip(methane, tmp_path, suffix):
    session = EditSession(methane)
    session.apply("replace_hydrogen", atom_id=20)
    mol = session.molecule
    path = str(tmp_path / f"edited.{suffix}")
    getattr(mol, f"to_{suffix}")(path)
    restored = getattr(Molecule, f"from_{suffix}")(path)
    assert restored.elements == mol.elements
    np.testing.assert_allclose(
        restored.atom_records["coords"], mol.atom_records["coords"], atol=0.0006
    )
    assert len(restored.bonds) == 7
    if suffix in ("npy", "hdf5"):
        assert restored.bonds.to_dict() == mol.bonds.to_dict()
        np.testing.assert_array_equal(
            restored.atom_records["atom_index"], mol.atom_records["atom_index"]
        )
    if suffix == "pdb":
        assert restored.bonds.status == "partial"
        assert set(restored.bonds.records["order"]) == {"unknown"}


def test_partial_mol2_roundtrip(methane, tmp_path):
    methane.bonds = BondGraph([(10, 20, "2", "manual")])
    path = str(tmp_path / "partial.mol2")
    methane.to_mol2(path)
    restored = Molecule.from_mol2(path)
    assert restored.bonds.status == "partial"
    assert restored.bonds.records["order"].tolist() == ["2"]


def test_frames_and_pop_preserve_graph(methane):
    combined = Molecule.cat_frames([methane, methane])
    assert combined.bonds.to_dict() == methane.bonds.to_dict()
    assert combined[1].bonds.to_dict() == methane.bonds.to_dict()
    with pytest.raises(ValueError, match="frame"):
        EditSession(combined)
    assert len(EditSession(combined, frame=1).molecule.atom_records) == 5
    other = methane[0]
    other.bonds = None
    with pytest.raises(ValueError, match="bond graphs"):
        Molecule.cat_frames([methane, other])
    methane.pop(1)
    assert methane.bonds.neighbors(10) == [30, 40, 50]
    methane.elements = ["N", "H", "H", "H"]
    assert methane.atom_records["atom_index"].tolist() == [10, 30, 40, 50]


def test_pubchem_bonds_use_atom_ids(monkeypatch):
    record = {
        "PC_Compounds": [
            {
                "atoms": {"aid": [9, 3], "element": [6, 8]},
                "bonds": {"aid1": [3], "aid2": [9], "order": [2]},
                "coords": [
                    {
                        "aid": [3, 9],
                        "conformers": [{"x": [1.2, 0], "y": [0, 0], "z": [0, 0]}],
                    }
                ],
            }
        ]
    }
    monkeypatch.setattr(
        pubchem, "urlopen", lambda *a, **kw: io.BytesIO(json.dumps(record).encode())
    )
    mol = Molecule.from_pubchem(123)
    assert mol.bonds.to_dict() == {
        "records": [(0, 1, "2", "imported")],
        "status": "complete",
    }
    np.testing.assert_allclose(mol.atom_records["coords"][0, 0], [0, 0, 0])


def test_pdb_edit_does_not_replay_raw_text(methane, tmp_path):
    path = str(tmp_path / "source.pdb")
    methane.to_pdb(path)
    original = Molecule.from_pdb(path)
    session = EditSession(original)
    session.apply("set_element", atom_id=1, element="F")
    text = session.molecule.to_pdb(path, return_str=True)
    assert text.splitlines()[1][76:78].strip() == "F"
    assert original.to_pdb(path, return_str=True).splitlines()[1][76:78].strip() == "H"


def test_widget_revisions_and_repeated_commands(methane):
    pytest.importorskip("anywidget")
    widget = methane.show(edit=True)
    message = {
        "id": "first",
        "revision": 0,
        "action": "set_element",
        "args": {"atom_id": 20, "element": "F"},
    }
    widget._handle_message(widget, message, [])
    assert widget.molecule.elements[1] == "F"
    assert widget.state["revision"] == 1
    widget._handle_message(widget, message, [])
    assert "already" in widget.state["error"]
    widget._handle_message(widget, {**message, "id": "stale"}, [])
    assert "changed" in widget.state["error"]
    assert widget.state["revision"] == 1
    widget.close()
