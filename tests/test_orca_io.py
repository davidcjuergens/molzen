"""ORCA stdout loading without companion geometry files."""

import numpy as np
import pytest

from molzen.io import Molecule


GEOMETRY = """---------------------------------
CARTESIAN COORDINATES (ANGSTROEM)
---------------------------------
  N     -1.199000   -1.399700    0.000000
  H      1.250300   -2.808200    0.000400

"""


def test_from_orca_stdout_results(tmp_path):
    path = tmp_path / "orca.out"
    path.write_text(
        GEOMETRY
        + """CARTESIAN COORDINATES (A.U.)
----------------------------
  NO LB      ZA    FRAG     MASS         X           Y           Z
   0 N     7.0000    0    14.007   -2.265782   -2.645050    0.000000
   1 H     1.0000    0     1.008    2.362725   -5.306729    0.000756

Number of atoms                             ...      2
 Total Charge           Charge          ....   -1
 Multiplicity           Mult            ....    3
FINAL SINGLE POINT ENERGY      -467.347015910351

CARTESIAN GRADIENT
------------------

   1   N   :    0.012815486    0.023015521   -0.000006123
   2   H   :   -0.007451532   -0.007331200   -0.000002136

Multiplicity       :   3
****ORCA TERMINATED NORMALLY****
"""
    )
    mol = Molecule.from_orca_stdout(path)
    assert mol.xyz.shape == (2, 3)
    assert mol.elements == ["N", "H"]
    np.testing.assert_array_equal(mol.Z, [7, 1])
    np.testing.assert_allclose(mol.xyz[0], [-1.199, -1.3997, 0])
    assert mol.spinmult == 3
    assert mol.metadata["orca"]["charge"] == -1
    assert mol.metadata["orca"]["terminated_normally"] is True
    (record,) = mol.excited_state_records
    assert record["frame_index"] == 0
    assert record["total_energy_au"] == -467.347015910351
    np.testing.assert_allclose(
        record["energy_gradient"],
        [
            [0.012815486, 0.023015521, -0.000006123],
            [-0.007451532, -0.007331200, -0.000002136],
        ],
    )


def test_multiple_geometries_align_results_and_support_slicing(tmp_path):
    path = tmp_path / "opt.out"
    path.write_text(
        GEOMETRY
        + "FINAL SINGLE POINT ENERGY -1.0D+02\n"
        + GEOMETRY.replace("-1.199000", "-1.000000")
        + "FINAL SINGLE POINT ENERGY -1.01e+02\n"
        + GEOMETRY.replace("-1.199000", "-1.000000")
    )
    mol = Molecule.from_orca_stdout(str(path))
    assert mol.xyz.shape == (3, 2, 3)
    assert [r["frame_index"] for r in mol.excited_state_records] == [0, 1]
    assert [r["total_energy_au"] for r in mol.excited_state_records] == [-100, -101]
    np.testing.assert_allclose(mol.xyz[:, 0, 0], [-1.199, -1, -1])
    sliced = mol.slice_frames(1, 3)
    assert sliced.excited_state_records[0]["frame_index"] == 0
    assert sliced.excited_state_records[0]["total_energy_au"] == -101


def test_geometry_only_and_eof(tmp_path):
    path = tmp_path / "unfinished.out"
    path.write_text(GEOMETRY.rstrip())
    mol = Molecule.from_orca_stdout(path)
    assert mol.xyz.shape == (2, 3)
    assert mol.spinmult is None
    assert mol.excited_state_records == []
    assert mol.metadata["orca"]["terminated_normally"] is False


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("ORCA failed before starting", "No Cartesian coordinates"),
        (GEOMETRY.replace("-1.199000", "broken"), "Invalid ORCA coordinate row"),
        (GEOMETRY.replace("-1.199000", "nan"), "Invalid ORCA coordinate row"),
        (GEOMETRY + GEOMETRY.replace("  N ", "  C "), "inconsistent atom"),
        (GEOMETRY + "Number of atoms ... 3\n", "atom count"),
        (GEOMETRY + "CARTESIAN GRADIENT\n---\n\n1 N : 0 0 0\n", "gradient atoms"),
        (GEOMETRY + "FINAL SINGLE POINT ENERGY broken\n", "Invalid ORCA final energy"),
        (GEOMETRY + "Multiplicity : 1\nMultiplicity : 3\n", "different spin"),
    ],
)
def test_invalid_output(tmp_path, text, message):
    path = tmp_path / "invalid.out"
    path.write_text(text)
    with pytest.raises(ValueError, match=message):
        Molecule.from_orca_stdout(path)


def test_missing_stdout(tmp_path):
    with pytest.raises(FileNotFoundError):
        Molecule.from_orca_stdout(tmp_path / "missing.out")
