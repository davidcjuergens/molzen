"""Tests for Molecule visualization integration."""

from __future__ import annotations

import sys
from builtins import __import__ as builtin_import
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from molzen.io.molecule import Molecule, atom_record_dtype
from molzen.constants import HARTREE2EV
from molzen.visualize import (
    _excited_state_energy_series,
    _excited_state_oscillator_strength_series,
)


class FakePy3DmolView:
    """Small fake py3Dmol view used for unit tests."""

    def __init__(self, *, width: str | int, height: str | int) -> None:
        self.width = width
        self.height = height
        self.startjs = (
            '<div id="3dmolviewer_UNIQUEID"  '
            'style="position: relative; width: 420px; height: 240px;">\n'
            "</div>\n<script>\n"
        )
        self.model_text: str | None = None
        self.model_format: str | None = None
        self.frames_text: str | None = None
        self.frames_format: str | None = None
        self.style = None
        self.style_calls = []
        self.animation_options = None
        self.zoomed = False

    def addModel(self, text: str, fmt: str) -> None:
        self.model_text = text
        self.model_format = fmt

    def addModelsAsFrames(self, text: str, fmt: str) -> None:
        self.frames_text = text
        self.frames_format = fmt

    def setStyle(self, *args) -> None:
        self.style_calls.append(args)
        self.style = args[-1]

    def animate(self, options) -> None:
        self.animation_options = options

    def zoomTo(self) -> None:
        self.zoomed = True


class FakePy3Dmol(ModuleType):
    """Small fake py3Dmol module used for unit tests."""

    def __init__(self) -> None:
        super().__init__("py3Dmol")
        self.last_view: FakePy3DmolView | None = None

    def view(self, *, width: str | int, height: str | int) -> FakePy3DmolView:
        self.last_view = FakePy3DmolView(width=width, height=height)
        return self.last_view


def test_show_raises_without_py3dmol(monkeypatch) -> None:
    mol = Molecule(
        xyz=np.array([[0.0, 0.0, 0.0]], dtype=float),
        elements=["H"],
    )

    monkeypatch.delitem(sys.modules, "py3Dmol", raising=False)

    def fake_import(name, *args, **kwargs):
        if name == "py3Dmol":
            raise ImportError("missing test dependency")
        return builtin_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(ImportError, match="py3Dmol is required"):
        mol.show()


def test_show_single_frame_xyz_molecule(monkeypatch) -> None:
    fake_py3dmol = FakePy3Dmol()
    monkeypatch.setitem(sys.modules, "py3Dmol", fake_py3dmol)

    mol = Molecule(
        xyz=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float),
        elements=["C", "O"],
    )

    view = mol.show(width="420px", height="240px")

    assert view is fake_py3dmol.last_view
    assert view.width == "420px"
    assert view.height == "240px"
    assert view.model_format == "xyz"
    assert view.frames_text is None
    assert view.model_text.startswith("2\nFrame 1\n")
    assert "C 0.00000000 0.00000000 0.00000000" in view.model_text
    assert "O 0.00000000 0.00000000 1.00000000" in view.model_text
    assert view.style_calls == [
        ({}, {"stick": {"radius": 0.12}, "sphere": {"scale": 0.25}}),
        ({"elem": "H"}, {"stick": {"radius": 0.06}, "sphere": {"scale": 0.21}}),
    ]
    assert "margin-left: auto; margin-right: auto" in view.startjs
    assert view.zoomed


def test_show_multiframe_molecule_uses_py3dmol_frames(monkeypatch) -> None:
    fake_py3dmol = FakePy3Dmol()
    monkeypatch.setitem(sys.modules, "py3Dmol", fake_py3dmol)

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

    assert view is fake_py3dmol.last_view
    assert view.model_text is None
    assert view.frames_format == "xyz"
    assert view.frames_text.count("2\nFrame") == 2
    assert "C 1.00000000 0.00000000 0.00000000" in view.frames_text
    assert "O 1.00000000 0.00000000 1.00000000" in view.frames_text
    assert view.animation_options is None
    assert view.style_calls == [
        ({}, {"stick": {"radius": 0.12}, "sphere": {"scale": 0.25}}),
        ({"elem": "H"}, {"stick": {"radius": 0.06}, "sphere": {"scale": 0.21}}),
    ]
    assert "width: 500px" in view.startjs
    assert "margin: 4px auto 0 auto" in view.startjs
    assert 'max="1"' in view.startjs
    assert "setFrame(frame)" in view.startjs
    assert view.zoomed


def test_show_can_limit_displayed_frame_range(monkeypatch) -> None:
    fake_py3dmol = FakePy3Dmol()
    monkeypatch.setitem(sys.modules, "py3Dmol", fake_py3dmol)

    mol = Molecule(
        xyz=np.array(
            [
                [[0.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0]],
                [[2.0, 0.0, 0.0]],
                [[3.0, 0.0, 0.0]],
            ],
            dtype=float,
        ),
        elements=["H"],
    )

    view = mol.show(start=1, end=3)

    assert view.frames_format == "xyz"
    assert view.frames_text.count("1\nFrame") == 2
    assert "H 0.00000000 0.00000000 0.00000000" not in view.frames_text
    assert "H 1.00000000 0.00000000 0.00000000" in view.frames_text
    assert "H 2.00000000 0.00000000 0.00000000" in view.frames_text
    assert "H 3.00000000 0.00000000 0.00000000" not in view.frames_text
    assert 'max="1"' in view.startjs


def test_show_multiframe_molecule_adds_excited_state_energy_plot(monkeypatch) -> None:
    fake_py3dmol = FakePy3Dmol()
    monkeypatch.setitem(sys.modules, "py3Dmol", fake_py3dmol)

    mol = Molecule(
        xyz=np.array(
            [
                [[0.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0]],
            ],
            dtype=float,
        ),
        elements=["H"],
        excited_state_records=[
            {
                "frame_index": 0,
                "state_j": 0,
                "multiplicity": "singlet",
                "total_energy_au": -10.0,
            },
            {
                "frame_index": 1,
                "state_j": 0,
                "multiplicity": "singlet",
                "total_energy_au": -11.0,
            },
            {
                "frame_index": 0,
                "state_i": 0,
                "state_j": 1,
                "multiplicity": "singlet",
                "total_energy_au": -9.0,
                "osc_strength": 0.1,
            },
            {
                "frame_index": 1,
                "state_i": 0,
                "state_j": 1,
                "multiplicity": "singlet",
                "total_energy_au": -9.5,
                "osc_strength": 0.2,
            },
        ],
    )

    view = mol.show()

    assert "molzen_energy_panel_UNIQUEID" in view.startjs
    assert "molzen_oscillator_panel_UNIQUEID" in view.startjs
    assert "molzen_show_outer_UNIQUEID" in view.startjs
    assert "display: inline-flex" in view.startjs
    assert "State energies rel. S0 frame 1 (eV)" in view.startjs
    assert "S0 -&gt; Sn oscillator strengths" in view.startjs
    assert "S0-&gt;S1" in view.startjs
    assert "molzen_plot_cursor_UNIQUEID" in view.startjs
    assert "window.molzenUpdateEnergyFrame_UNIQUEID" in view.startjs
    assert "molzenUpdateEnergyFrame_UNIQUEID(frame)" in view.startjs
    assert 'height="300"' in view.startjs
    assert "S0" in view.startjs
    assert "S1" in view.startjs
    assert 'color: #c2410c; white-space: nowrap;">S0-&gt;S1' in view.startjs


def test_show_marks_cat_frame_boundaries_in_state_plots(monkeypatch) -> None:
    fake_py3dmol = FakePy3Dmol()
    monkeypatch.setitem(sys.modules, "py3Dmol", fake_py3dmol)

    mol_a = Molecule(
        xyz=np.array(
            [
                [[0.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0]],
            ],
            dtype=float,
        ),
        elements=["H"],
        excited_state_records=[
            {"frame_index": 0, "state_j": 0, "total_energy_au": -10.0},
            {"frame_index": 1, "state_j": 0, "total_energy_au": -9.0},
        ],
    )
    mol_b = Molecule(
        xyz=np.array([[[2.0, 0.0, 0.0]]], dtype=float),
        elements=["H"],
        excited_state_records=[
            {"frame_index": 0, "state_j": 0, "total_energy_au": -8.0},
        ],
    )
    mol = Molecule.cat_frames([mol_a, mol_b])

    view = mol.show()

    assert mol.metadata["cat_frames"]["frame_boundaries"] == [2]
    assert "molzen_cat_boundary_UNIQUEID" in view.startjs
    assert 'stroke-dasharray="4 4"' in view.startjs
    assert 'stroke="#9ca3af"' in view.startjs


def test_excited_state_energy_series_converts_hartree_to_relative_ev() -> None:
    series = _excited_state_energy_series(
        [
            {
                "frame_index": 0,
                "state_j": 0,
                "multiplicity": "singlet",
                "total_energy_au": -2.0,
            },
            {
                "frame_index": 0,
                "state_j": 1,
                "multiplicity": "singlet",
                "total_energy_au": -1.5,
            },
        ]
    )

    assert series[0]["y"] == [0.0]
    assert series[1]["y"] == pytest.approx([0.5 * HARTREE2EV])


def test_excited_state_oscillator_strength_series_uses_s0_to_sn_records() -> None:
    series = _excited_state_oscillator_strength_series(
        [
            {"frame_index": 0, "state_i": 0, "state_j": 0, "osc_strength": 0.0},
            {
                "frame_index": 0,
                "state_i": 0,
                "state_j": 1,
                "multiplicity": "singlet",
                "osc_strength": 0.25,
            },
        ]
    )

    assert series == [{"label": "S0->S1", "x": [0], "y": [0.25], "color_index": 1}]


def test_show_explicit_frame(monkeypatch) -> None:
    fake_py3dmol = FakePy3Dmol()
    monkeypatch.setitem(sys.modules, "py3Dmol", fake_py3dmol)

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

    assert view.model_format == "xyz"
    assert view.frames_text is None
    assert view.animation_options is None
    assert "O 0.00000000 0.00000000 1.00000000" in view.model_text
    assert "O 5.00000000 0.00000000 1.00000000" not in view.model_text


def test_show_out_of_range_frame_raises(monkeypatch) -> None:
    fake_py3dmol = FakePy3Dmol()
    monkeypatch.setitem(sys.modules, "py3Dmol", fake_py3dmol)

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
    fake_py3dmol = FakePy3Dmol()
    monkeypatch.setitem(sys.modules, "py3Dmol", fake_py3dmol)

    pdb_file = Path(__file__).parent / "data" / "1PRW.pdb"
    mol = Molecule.from_pdb(str(pdb_file))

    view = mol.show(frame=0)

    assert view.model_format == "xyz"
    assert view.model_text.startswith(f"{len(mol.atom_records)}\nFrame 1\n")
    assert "C 56.83300018 25.00600052 2.16499996" in view.model_text


def test_show_hetatm_only_molecule_contains_hetatm(monkeypatch) -> None:
    fake_py3dmol = FakePy3Dmol()
    monkeypatch.setitem(sys.modules, "py3Dmol", fake_py3dmol)

    mol = Molecule(
        xyz=np.array([[0.0, 0.0, 0.0]], dtype=float),
        elements=["Cl"],
    )

    view = mol.show()

    assert view.model_format == "xyz"
    assert view.model_text.startswith("1\nFrame 1\n")
    assert "Cl 0.00000000 0.00000000 0.00000000" in view.model_text


def test_show_normalizes_lowercase_hydrogen_element(monkeypatch) -> None:
    fake_py3dmol = FakePy3Dmol()
    monkeypatch.setitem(sys.modules, "py3Dmol", fake_py3dmol)

    mol = Molecule(
        xyz=np.array([[0.0, 0.0, 0.0]], dtype=float),
        elements=["h"],
    )

    view = mol.show()

    assert "H 0.00000000 0.00000000 0.00000000" in view.model_text
    assert view.style_calls[-1] == (
        {"elem": "H"},
        {"stick": {"radius": 0.06}, "sphere": {"scale": 0.21}},
    )


def test_show_uses_canonical_coordinates_after_mutation(monkeypatch) -> None:
    fake_py3dmol = FakePy3Dmol()
    monkeypatch.setitem(sys.modules, "py3Dmol", fake_py3dmol)

    pdb_file = Path(__file__).parent / "data" / "1PRW.pdb"
    mol = Molecule.from_pdb(str(pdb_file))

    xyz = np.asarray(mol.xyz, dtype=float).copy()
    xyz[0] = np.array([123.456, 78.9, -10.111], dtype=float)
    mol.xyz = xyz

    view = mol.show(frame=0)

    assert "C 123.45600128 78.90000153 -10.11100006" in view.model_text


def test_show_empty_molecule_raises_value_error(monkeypatch) -> None:
    fake_py3dmol = FakePy3Dmol()
    monkeypatch.setitem(sys.modules, "py3Dmol", fake_py3dmol)

    mol = Molecule(atom_records=np.zeros(0, dtype=atom_record_dtype(1)))

    with pytest.raises(ValueError, match="Cannot visualize an empty molecule"):
        mol.show()
