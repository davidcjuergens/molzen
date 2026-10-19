"""Geometry measurements and the shared viewer panel."""

from pathlib import Path

import numpy as np
import pytest

from molzen.geometry import bond_angle, describe_geometry, dihedral_angle, distance


def test_distance_angle_and_signed_dihedral():
    assert distance([0.0, 0.0, 0.0], [0.0, 0.0, 2.5]) == pytest.approx(2.5)
    assert bond_angle(
        [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]
    ) == pytest.approx(90.0)
    assert dihedral_angle(
        [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0]
    ) == pytest.approx(90.0)
    assert dihedral_angle(
        [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, -1.0]
    ) == pytest.approx(-90.0)
    assert dihedral_angle(
        [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, -1.0, 0.0]
    ) == pytest.approx(180.0)


def test_describe_geometry_reports_each_selection_size():
    coordinates = describe_geometry([[1.25, -2.0, 0.5]])
    assert coordinates == {"kind": "coordinates", "value": [1.25, -2.0, 0.5]}

    separation = describe_geometry([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    assert separation["kind"] == "distance"
    assert separation["value"] == pytest.approx(3.0)

    angle = describe_geometry([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert angle["kind"] == "angle"
    assert angle["value"] == pytest.approx(90.0)

    torsion = describe_geometry(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0]]
    )
    assert torsion["kind"] == "dihedral"
    assert torsion["value"] == pytest.approx(90.0)


def test_describe_geometry_rejects_bad_input_and_undefined_selections():
    with pytest.raises(ValueError, match="1 to 4"):
        describe_geometry(np.zeros((5, 3)))
    assert describe_geometry([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])["kind"] == "undefined"
    assert (
        describe_geometry(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 1.0, 0.0]]
        )["kind"]
        == "undefined"
    )


def test_geometry_panel_is_shared_by_the_editor_and_show():
    root = Path(__file__).parents[1]
    panel = (root / "molzen/editor/static/geometry_panel.js").read_text()
    editor = (root / "molzen/editor/static/editor.js").read_text()
    widget = (root / "molzen/editor/__init__.py").read_text()
    viewer = (root / "molzen/visualize.py").read_text()

    assert "molzenGeometry" in panel
    assert "Select 1–4 atoms." in panel
    assert "molzenGeometry.render" in editor
    assert "molzenGeometry.maxAtoms" in editor
    assert "geometry_panel.js" in widget
    assert "geometry_panel.js" in viewer
    assert "_add_py3dmol_geometry_panel" in viewer
