"""Tests for dependency-free DCD parsing."""

from __future__ import annotations

import struct

import numpy as np

from molzen.io.dcd import parse_dcd_with_prmtop
from molzen.io.molecule import Molecule


def _write_record(f, data: bytes) -> None:
    f.write(struct.pack("<i", len(data)))
    f.write(data)
    f.write(struct.pack("<i", len(data)))


def _write_dcd(path, coords: np.ndarray) -> None:
    coords = np.asarray(coords, dtype=np.float32)
    n_frames, n_atoms, _ = coords.shape
    header_ints = [0] * 20
    header_ints[0] = n_frames
    header_ints[1] = 0
    header_ints[2] = 1

    with open(path, "wb") as f:
        _write_record(f, b"CORD" + struct.pack("<20i", *header_ints))
        _write_record(f, struct.pack("<i", 1) + b"unit-test dcd".ljust(80))
        _write_record(f, struct.pack("<i", n_atoms))
        for frame in coords:
            for axis in range(3):
                _write_record(f, np.ascontiguousarray(frame[:, axis]).tobytes())


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


def test_parse_dcd_with_prmtop_selects_zero_based_qmindices(tmp_path):
    dcd_path = tmp_path / "optim.dcd"
    prmtop_path = tmp_path / "system.prmtop"
    qmindices_path = tmp_path / "qm.qm"
    coords = np.arange(30, dtype=float).reshape(2, 5, 3)
    _write_dcd(dcd_path, coords)
    _write_prmtop(prmtop_path)
    qmindices_path.write_text("0 4\n")

    payload = parse_dcd_with_prmtop(
        str(dcd_path),
        str(prmtop_path),
        qmindices_fp=str(qmindices_path),
    )

    np.testing.assert_allclose(payload["xyz"], coords[:, [0, 4], :])
    assert payload["atom_names"] == ["N", "O"]
    assert payload["elements"] == ["N", "O"]
    assert payload["serials"] == [1, 5]
    assert payload["metadata"]["source_atom_indices"] == [0, 4]
    assert payload["metadata"]["dcd_n_frames"] == 2


def test_molecule_from_dcd_preserves_frames_and_amber_labels(tmp_path):
    dcd_path = tmp_path / "optim.dcd"
    prmtop_path = tmp_path / "system.prmtop"
    qmindices_path = tmp_path / "qm.qm"
    coords = np.arange(30, dtype=float).reshape(2, 5, 3)
    _write_dcd(dcd_path, coords)
    _write_prmtop(prmtop_path)
    qmindices_path.write_text("1 3\n")

    mol = Molecule.from_dcd(
        str(dcd_path),
        str(prmtop_path),
        qmindices_path=str(qmindices_path),
    )

    np.testing.assert_allclose(mol.xyz, coords[:, [1, 3], :])
    assert mol.atom_records["atom_name"].tolist() == ["H", "C"]
    assert mol.atom_records["res_name"].tolist() == ["ALA", "ALA"]
    assert mol.atom_records["serial"].tolist() == [2, 4]
    assert mol.metadata["dcd_n_frames"] == 2


def test_molecule_from_terachem_stdout_finds_optimization_dcd(tmp_path):
    scrdir = tmp_path / "tc_scr.case"
    scrdir.mkdir()
    dcd_path = scrdir / "optim.dcd"
    prmtop_path = tmp_path / "system.prmtop"
    qmindices_path = tmp_path / "qm.qm"
    stdout_path = tmp_path / "job.out"
    coords = np.arange(30, dtype=float).reshape(2, 5, 3)
    _write_dcd(dcd_path, coords)
    _write_prmtop(prmtop_path)
    qmindices_path.write_text("0 4\n")

    # These fallback files should not make the result ambiguous when optim.dcd exists.
    (scrdir / "optim.xyz").write_text("1\nframe\nH 0.0 0.0 0.0\n")
    (scrdir / "optim.rst7").write_text("not parsed when dcd exists\n")
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

    np.testing.assert_allclose(mol.xyz, coords[:, [0, 4], :])
    assert mol.atom_records["atom_name"].tolist() == ["N", "O"]
    assert mol.metadata["dcd_path"] == str(dcd_path)
    assert mol.metadata["terachem_input_args"]["run"] == "minimize"


def test_molecule_from_terachem_stdout_discovers_scrdir_from_stdout_tag(tmp_path):
    tag = "optim_design_frame_015"
    scrdir = tmp_path / f"tc_scr.{tag}"
    scrdir.mkdir()
    dcd_path = scrdir / "optim.dcd"
    prmtop_path = tmp_path / "system.prmtop"
    qmindices_path = tmp_path / "qm.qm"
    stdout_path = tmp_path / f"stdout_{tag}.log"
    coords = np.arange(30, dtype=float).reshape(2, 5, 3)
    _write_dcd(dcd_path, coords)
    _write_prmtop(prmtop_path)
    qmindices_path.write_text("0 4\n")
    stdout_path.write_text(
        "\n".join(
            [
                "Processed Input file:",
                f"scrdir /old-cluster/tc_scr.{tag}",
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

    np.testing.assert_allclose(mol.xyz, coords[:, [0, 4], :])
    assert mol.metadata["dcd_path"] == str(dcd_path)
    assert mol.metadata["terachem_input_args"]["scrdir"] == (
        f"/old-cluster/tc_scr.{tag}"
    )


def test_molecule_from_terachem_stdout_accepts_dcd_path_overrides(tmp_path):
    dcd_path = tmp_path / "moved.dcd"
    prmtop_path = tmp_path / "moved.prmtop"
    qmindices_path = tmp_path / "moved.qm"
    stdout_path = tmp_path / "job.out"
    coords = np.arange(30, dtype=float).reshape(2, 5, 3)
    _write_dcd(dcd_path, coords)
    _write_prmtop(prmtop_path)
    qmindices_path.write_text("1 3\n")
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
        dcd_path=str(dcd_path),
        prmtop_path=str(prmtop_path),
        qmindices_path=str(qmindices_path),
    )

    np.testing.assert_allclose(mol.xyz, coords[:, [1, 3], :])
    assert mol.atom_records["atom_name"].tolist() == ["H", "C"]
    assert mol.metadata["dcd_path"] == str(dcd_path)
    assert mol.metadata["prmtop_path"] == str(prmtop_path)
    assert mol.metadata["qmindices_path"] == str(qmindices_path)
