"""Tests for Amber restart/prmtop parsing."""

from __future__ import annotations

import numpy as np

from molzen.io.molecule import Molecule
from molzen.io.rst7 import parse_qmindices, parse_rst7_with_prmtop


def _write_rst7(path, coords: np.ndarray) -> None:
    values = coords.reshape(-1)
    lines = ["unit-test restart\n", f"{coords.shape[0]:6d}\n"]
    for i in range(0, len(values), 6):
        lines.append("".join(f"{value:12.7f}" for value in values[i : i + 6]) + "\n")
    path.write_text("".join(lines))


def _write_prmtop(path) -> None:
    path.write_text(
        "\n".join(
            [
                "%VERSION  VERSION_STAMP = V0001.000  DATE = 01/01/01  00:00:00",
                "%FLAG ATOM_NAME",
                "%FORMAT(20a4)",
                "N   H   CA  C   O   ",
                "%FLAG AMBER_ATOM_TYPE",
                "%FORMAT(20a4)",
                "N   H   CT  C   O   ",
                "%FLAG ATOMIC_NUMBER",
                "%FORMAT(10I8)",
                f"{7:8d}{1:8d}{6:8d}{6:8d}{8:8d}",
                "%FLAG RESIDUE_LABEL",
                "%FORMAT(20a4)",
                "ALA WAT ",
                "%FLAG RESIDUE_POINTER",
                "%FORMAT(10I8)",
                f"{1:8d}{5:8d}",
                "",
            ]
        )
    )


def test_parse_rst7_with_prmtop_selects_zero_based_qmindices(tmp_path):
    rst7_path = tmp_path / "optim.rst7"
    prmtop_path = tmp_path / "system.prmtop"
    qmindices_path = tmp_path / "qm.qm"
    coords = np.asarray(
        [
            [0.0, 0.1, 0.2],
            [1.0, 1.1, 1.2],
            [2.0, 2.1, 2.2],
            [3.0, 3.1, 3.2],
            [4.0, 4.1, 4.2],
        ],
        dtype=float,
    )
    _write_rst7(rst7_path, coords)
    _write_prmtop(prmtop_path)
    qmindices_path.write_text("0 2 4\n")

    payload = parse_rst7_with_prmtop(
        str(rst7_path),
        str(prmtop_path),
        qmindices_fp=str(qmindices_path),
    )

    np.testing.assert_allclose(payload["xyz"], coords[[0, 2, 4]])
    assert payload["atom_names"] == ["N", "CA", "O"]
    assert payload["elements"] == ["N", "C", "O"]
    assert payload["res_names"] == ["ALA", "ALA", "WAT"]
    assert payload["res_nums"] == [1, 1, 2]
    assert payload["serials"] == [1, 3, 5]
    assert payload["metadata"]["source_atom_indices"] == [0, 2, 4]


def test_parse_qmindices_accepts_explicit_zero_based_indices(tmp_path):
    qmindices_path = tmp_path / "qm.qm"
    qmindices_path.write_text("0\n2\n")

    np.testing.assert_array_equal(parse_qmindices(str(qmindices_path)), [0, 2])


def test_molecule_from_rst7_preserves_amber_labels(tmp_path):
    rst7_path = tmp_path / "optim.rst7"
    prmtop_path = tmp_path / "system.prmtop"
    qmindices_path = tmp_path / "qm.qm"
    coords = np.arange(15, dtype=float).reshape(5, 3)
    _write_rst7(rst7_path, coords)
    _write_prmtop(prmtop_path)
    qmindices_path.write_text("0 2 4\n")

    mol = Molecule.from_rst7(
        str(rst7_path),
        str(prmtop_path),
        qmindices_path=str(qmindices_path),
    )

    np.testing.assert_allclose(mol.xyz, coords[[0, 2, 4]])
    assert mol.atom_records["atom_name"].tolist() == ["N", "CA", "O"]
    assert mol.atom_records["element"].tolist() == ["N", "C", "O"]
    assert mol.atom_records["res_name"].tolist() == ["ALA", "ALA", "WAT"]
    assert mol.atom_records["res_num"].tolist() == [1, 1, 2]
    assert mol.atom_records["serial"].tolist() == [1, 3, 5]


def test_molecule_from_terachem_stdout_finds_optimization_rst7(tmp_path):
    scrdir = tmp_path / "tc_scr.case"
    scrdir.mkdir()
    rst7_path = scrdir / "optim.rst7"
    prmtop_path = tmp_path / "system.prmtop"
    qmindices_path = tmp_path / "qm.qm"
    stdout_path = tmp_path / "job.out"
    coords = np.arange(15, dtype=float).reshape(5, 3)
    _write_rst7(rst7_path, coords)
    _write_prmtop(prmtop_path)
    qmindices_path.write_text("1 3\n")
    stdout_path.write_text(
        "\n".join(
            [
                "Processed Input file:",
                f"scrdir {scrdir}",
                "run minimize",
                f"coordinates {tmp_path / 'input.rst7'}",
                f"prmtop {prmtop_path}",
                f"qmindices {qmindices_path}",
                "spinmult 1",
                "---------------------",
                "",
            ]
        )
    )

    mol = Molecule.from_terachem_stdout(str(stdout_path))

    np.testing.assert_allclose(mol.xyz, coords[[1, 3]])
    assert mol.atom_records["atom_name"].tolist() == ["H", "C"]
    assert mol.atom_records["serial"].tolist() == [2, 4]
    assert mol.metadata["terachem_input_args"]["run"] == "minimize"
    assert mol.metadata["source_atom_indices"] == [1, 3]


def test_molecule_from_terachem_stdout_rejects_ambiguous_optimization_files(
    tmp_path,
):
    scrdir = tmp_path / "tc_scr.case"
    scrdir.mkdir()
    _write_rst7(scrdir / "optim.rst7", np.arange(15, dtype=float).reshape(5, 3))
    (scrdir / "optim.xyz").write_text("1\nframe\nH 0.0 0.0 0.0\n")
    prmtop_path = tmp_path / "system.prmtop"
    _write_prmtop(prmtop_path)
    stdout_path = tmp_path / "job.out"
    stdout_path.write_text(
        "\n".join(
            [
                "Processed Input file:",
                f"scrdir {scrdir}",
                "run minimize",
                f"coordinates {tmp_path / 'input.rst7'}",
                f"prmtop {prmtop_path}",
                "spinmult 1",
                "---------------------",
                "",
            ]
        )
    )

    with np.testing.assert_raises_regex(
        ValueError,
        "found both",
    ):
        Molecule.from_terachem_stdout(str(stdout_path))


def test_molecule_from_terachem_stdout_accepts_rst7_path_overrides(tmp_path):
    rst7_path = tmp_path / "moved.rst7"
    prmtop_path = tmp_path / "moved.prmtop"
    qmindices_path = tmp_path / "moved.qm"
    stdout_path = tmp_path / "job.out"
    coords = np.arange(15, dtype=float).reshape(5, 3)
    _write_rst7(rst7_path, coords)
    _write_prmtop(prmtop_path)
    qmindices_path.write_text("0 4\n")
    stdout_path.write_text(
        "\n".join(
            [
                "Processed Input file:",
                "scrdir /old-cluster/tc_scr.case",
                "run minimize",
                "coordinates /old-cluster/input.rst7",
                "prmtop /old-cluster/system.prmtop",
                "qmindices /old-cluster/qm.qm",
                "spinmult 1",
                "---------------------",
                "",
            ]
        )
    )

    mol = Molecule.from_terachem_stdout(
        str(stdout_path),
        rst7_path=str(rst7_path),
        prmtop_path=str(prmtop_path),
        qmindices_path=str(qmindices_path),
    )

    np.testing.assert_allclose(mol.xyz, coords[[0, 4]])
    assert mol.atom_records["atom_name"].tolist() == ["N", "O"]
    assert mol.metadata["rst7_path"] == str(rst7_path)
    assert mol.metadata["prmtop_path"] == str(prmtop_path)
    assert mol.metadata["qmindices_path"] == str(qmindices_path)
    assert mol.metadata["terachem_input_args"]["scrdir"] == "/old-cluster/tc_scr.case"
