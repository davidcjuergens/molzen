"""Tests for Molecule nglview integration."""

from __future__ import annotations

import sys
from builtins import __import__ as builtin_import
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from molzen.io.molecule import Molecule, atom_record_dtype


class FakeView:
    """Small fake nglview widget used for unit tests."""

    def __init__(self, text: str) -> None:
        self.show_text_input = text
        self.ball_and_stick_called = False
        self.layout = SimpleNamespace(width=None, height=None)

    def add_ball_and_stick(self) -> None:
        self.ball_and_stick_called = True


class FakeStructure:
    """Small fake nglview.Structure base class."""

    def __init__(self) -> None:
        self.ext = "pdb"
        self.params = {}


class FakeTrajectory:
    """Small fake nglview.Trajectory base class."""

    def __init__(self) -> None:
        self.shown = True


class FakeNGLView(ModuleType):
    """Small fake nglview module used for unit tests."""

    def __init__(self) -> None:
        super().__init__("nglview")
        self.Structure = FakeStructure
        self.Trajectory = FakeTrajectory
        self.NGLWidget = self._make_widget
        self.last_view: FakeView | None = None
        self.last_widget_input = None
        self.last_widget_kwargs = None

    def show_text(self, text: str) -> FakeView:
        self.last_view = FakeView(text)
        return self.last_view

    def _make_widget(self, structure, **kwargs) -> FakeView:
        view = FakeView(structure.get_structure_string())
        view.structure = structure
        view.gui_style = None
        self.last_view = view
        self.last_widget_input = structure
        self.last_widget_kwargs = kwargs
        return view


def test_show_raises_without_nglview(monkeypatch) -> None:
    mol = Molecule(
        xyz=np.array([[0.0, 0.0, 0.0]], dtype=float),
        elements=["H"],
    )

    monkeypatch.delitem(sys.modules, "nglview", raising=False)

    def fake_import(name, *args, **kwargs):
        if name == "nglview":
            raise ImportError("missing test dependency")
        return builtin_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(ImportError, match="nglview is required for Molecule.show"):
        mol.show()


def test_show_single_frame_xyz_molecule(monkeypatch) -> None:
    fake_nv = FakeNGLView()
    monkeypatch.setitem(sys.modules, "nglview", fake_nv)

    mol = Molecule(
        xyz=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float),
        elements=["C", "O"],
    )

    view = mol.show(width="420px", height="240px")

    assert view is fake_nv.last_view
    assert "MODEL" not in view.show_text_input
    assert "HETATM" in view.show_text_input
    assert "  0.000   0.000   1.000" in view.show_text_input
    assert "420px" in view.layout.width
    assert "240px" in view.layout.height
    assert view.ball_and_stick_called is True


def test_show_multiframe_molecule_uses_trajectory_widget(monkeypatch) -> None:
    fake_nv = FakeNGLView()
    monkeypatch.setitem(sys.modules, "nglview", fake_nv)

    mol = Molecule(
        xyz=np.array(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0], [1.0, 0.0, 1.0]],
            ],
            dtype=float,
        ),
        elements=["C", "O"],
    )

    view = mol.show()

    assert fake_nv.last_widget_input is not None
    assert fake_nv.last_widget_kwargs == {"gui": True}

    assert fake_nv.last_widget_input.n_frames == 2
    np.testing.assert_allclose(
        fake_nv.last_widget_input.get_coordinates(1),
        np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 1.0]], dtype=float),
    )
    assert "MODEL" not in view.show_text_input


def test_show_explicit_frame(monkeypatch) -> None:
    fake_nv = FakeNGLView()
    monkeypatch.setitem(sys.modules, "nglview", fake_nv)

    mol = Molecule(
        xyz=np.array(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                [[5.0, 0.0, 0.0], [5.0, 0.0, 1.0]],
            ],
            dtype=float,
        ),
        elements=["C", "O"],
    )

    view = mol.show(frame=0)

    assert "MODEL" not in view.show_text_input
    assert "  0.000   0.000   1.000" in view.show_text_input
    assert "  5.000   0.000   1.000" not in view.show_text_input


def test_show_out_of_range_frame_raises(monkeypatch) -> None:
    fake_nv = FakeNGLView()
    monkeypatch.setitem(sys.modules, "nglview", fake_nv)

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

    with pytest.raises(IndexError, match="Frame index 999 out of range"):
        mol.show(frame=999)


def test_show_polymer_pdb_contains_atom_records(monkeypatch) -> None:
    fake_nv = FakeNGLView()
    monkeypatch.setitem(sys.modules, "nglview", fake_nv)

    pdb_file = Path(__file__).parent / "data" / "1PRW.pdb"
    mol = Molecule.from_pdb(str(pdb_file))

    view = mol.show(frame=0)

    assert "ATOM  " in view.show_text_input


def test_show_hetatm_only_molecule_contains_hetatm(monkeypatch) -> None:
    fake_nv = FakeNGLView()
    monkeypatch.setitem(sys.modules, "nglview", fake_nv)

    mol = Molecule(
        xyz=np.array([[0.0, 0.0, 0.0]], dtype=float),
        elements=["Cl"],
    )

    view = mol.show()

    assert "HETATM" in view.show_text_input
    assert "ATOM  " not in view.show_text_input


def test_show_uses_canonical_coordinates_after_mutation(monkeypatch) -> None:
    fake_nv = FakeNGLView()
    monkeypatch.setitem(sys.modules, "nglview", fake_nv)

    pdb_file = Path(__file__).parent / "data" / "1PRW.pdb"
    mol = Molecule.from_pdb(str(pdb_file))

    xyz = np.asarray(mol.xyz, dtype=float).copy()
    xyz[0] = np.array([123.456, 78.9, -10.111], dtype=float)
    mol.xyz = xyz

    view = mol.show(frame=0)

    assert " 123.456  78.900 -10.111" in view.show_text_input


def test_show_empty_molecule_raises_value_error(monkeypatch) -> None:
    fake_nv = FakeNGLView()
    monkeypatch.setitem(sys.modules, "nglview", fake_nv)

    mol = Molecule(atom_records=np.zeros(0, dtype=atom_record_dtype(1)))

    with pytest.raises(ValueError, match="Cannot visualize an empty molecule"):
        mol.show()
