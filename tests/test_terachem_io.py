"""Test terachem output parsing"""

import math
import os

import numpy as np
import pytest

from molzen.io.terachem.parse import parse_terachem_output

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.local_only
def test_parse_casscf_optimization():
    """Tests the parsing of a CASSCF optimization output file from terachem."""
    output_path = os.path.join(THIS_DIR, "data/terachem_casscf_s0_opt.out")
    parsed = parse_terachem_output(output_path)

    # energies
    assert "casscf_energies" in parsed
    assert len(parsed["casscf_energies"]) == 4
    assert parsed["casscf_energies"][1]["total_energy_au"][0] == -1954.15973814
    assert parsed["casscf_energies"][2]["total_energy_au"][0] == -1954.01500372
    assert parsed["casscf_energies"][3]["total_energy_au"][0] == -1953.98558982
    assert parsed["casscf_energies"][4]["total_energy_au"][0] == -1953.97030442

    assert len(parsed["casscf_energies"][1]["total_energy_au"]) == 357

    # Transition dipoles
    assert "casscf_transition_dipoles" in parsed
    assert set(parsed["casscf_transition_dipoles"].keys()) == {"singlet"}
    assert len(parsed["casscf_transition_dipoles"]["singlet"]) == 6
    assert (
        len(parsed["casscf_transition_dipoles"]["singlet"]["1 -> 2"]["osc_strength"])
        == 357
    )

    # orbitals
    assert "casscf_orbitals" in parsed
    assert len(parsed["casscf_orbitals"]) == 8


@pytest.mark.local_only
def test_parse_terachem_output():
    """Tests parsing of terachem output file"""
    example_output_path = os.path.join(THIS_DIR, "data/terachem_example_output.out")
    parsed = parse_terachem_output(example_output_path)

    # check parsed inputs correctly
    """
                  coordinates ./ala.xyz
                      run energy
                    basis 6-31g**
                   method wb97xd3
                 convthre 1e-7
                  threall 1e-14
                   charge 0
                 spinmult 1
                   purify no
                      cis yes
                  cismult 1
             cisnumstates 10
             cisguessvecs 10
               cismaxiter 100
                  cisntos yes
                cistarget 1
                      pcm cosmo
                  epsilon 78.4
         ss_pcm_solvation neq
                    maxit 200
                     gpus 1
                   gpumem 256

    """
    assert parsed["input_args"]["coordinates"] == "./ala.xyz"
    assert parsed["input_args"]["run"] == "energy"
    assert parsed["input_args"]["basis"] == "6-31g**"
    assert parsed["input_args"]["method"] == "wb97xd3"
    assert parsed["input_args"]["convthre"] == "1e-7"
    assert parsed["input_args"]["threall"] == "1e-14"
    assert parsed["input_args"]["charge"] == "0"
    assert parsed["input_args"]["spinmult"] == "1"
    assert parsed["input_args"]["purify"] == "no"
    assert parsed["input_args"]["cis"] == "yes"
    assert parsed["input_args"]["cismult"] == "1"
    assert parsed["input_args"]["cisnumstates"] == "10"
    assert parsed["input_args"]["cisguessvecs"] == "10"
    assert parsed["input_args"]["cismaxiter"] == "100"
    assert parsed["input_args"]["cisntos"] == "yes"
    assert parsed["input_args"]["cistarget"] == "1"
    assert parsed["input_args"]["pcm"] == "cosmo"
    assert parsed["input_args"]["epsilon"] == "78.4"
    assert parsed["input_args"]["ss_pcm_solvation"] == "neq"
    assert parsed["input_args"]["maxit"] == "200"
    assert parsed["input_args"]["gpus"] == "1"
    assert parsed["input_args"]["gpumem"] == "256"

    es_entry = 5
    root = 1

    assert (
        parsed["excited_states"][es_entry - 1][root]["total_energy_au"] == -248.27323993
    )
    assert parsed["excited_states"][es_entry - 1][root]["exc_energy_ev"] == 4.04505765
    assert parsed["excited_states"][es_entry - 1][root]["osc_strength"] == 0.0030
    assert parsed["excited_states"][es_entry - 1][root]["s_squared"] == 0.0000
    assert parsed["excited_states"][es_entry - 1][root]["max_ci_coeff"] == -0.737191

    # should be 5 total entries
    assert len(parsed["excited_states"]) == 5

    # should be 10 excited states parsed for this entry
    assert len(parsed["excited_states"][es_entry - 1]) == 10

    root = 10
    assert (
        parsed["excited_states"][es_entry - 1][root]["total_energy_au"] == -248.05883516
    )
    assert parsed["excited_states"][es_entry - 1][root]["exc_energy_ev"] == 9.87930744
    assert parsed["excited_states"][es_entry - 1][root]["osc_strength"] == 0.0014
    assert parsed["excited_states"][es_entry - 1][root]["s_squared"] == 0.0000
    assert parsed["excited_states"][es_entry - 1][root]["max_ci_coeff"] == 0.639490


def test_parse_hhtda_superset_casscf_energy_header():
    """HH-TDA excited-state table should parse via the CASSCF-like energy parser."""
    raw = """
  Root   Mult.   Total Energy (a.u.)   Ex. Energy (a.u.)     Ex. Energy (eV)     Ex. Energy (nm)        Osc. (a.u.)
--------------------------------------------------------------------------------------------------------------------
   1 singlet -248.42161559 0.00000000 0.00000000 inf 0.00000000
   2 singlet -248.27323993 0.14837566 4.03751706 307.08688619 0.00300000
"""
    parsed = parse_terachem_output(raw, raw_str_in=True)

    assert "casscf_energies" in parsed
    assert parsed["casscf_energies"][1]["total_energy_au"] == [-248.42161559]
    assert math.isinf(parsed["casscf_energies"][1]["exc_energy_nm"][0])
    assert parsed["casscf_energies"][1]["osc_strength"] == [0.0]
    assert parsed["casscf_energies"][2]["exc_energy_au"] == [0.14837566]
    assert parsed["casscf_energies"][2]["exc_energy_ev"] == [4.03751706]
    assert parsed["casscf_energies"][2]["exc_energy_nm"] == [307.08688619]
    assert parsed["casscf_energies"][2]["osc_strength"] == [0.003]

    records = parsed["excited_state_records"]
    assert len(records) == 2
    assert records[0]["source"] == "parse_casscf_excited_state_section"
    assert records[0]["state_i"] == 0  # terachem hhtda/casscf-like tables are 1-indexed
    assert records[0]["state_j"] == 0  # ground state
    assert records[1]["exc_energy_nm"] == 307.08688619


def test_parse_final_excited_state_results_infers_multiplicity_from_s_squared():
    raw = """
  Final Excited State Results:

  Root   Total Energy (a.u.)   Ex. Energy (eV)   Osc. (a.u.)   < S^2 >   Max CI Coeff.      Excitation
------------------------------------------------------------------------------------------------------------
     1          -1.00000000        2.00000000       0.1000      0.0000      0.700000     1 ->   2 : D -> V
     2          -0.90000000        3.00000000       0.2000      2.0000      0.800000     1 ->   3 : D -> V

"""
    parsed = parse_terachem_output(raw, raw_str_in=True)

    records = parsed["excited_state_records"]
    assert records[0]["multiplicity"] == "singlet"
    assert records[1]["multiplicity"] == "triplet"


def test_parse_gradient_run_adds_energy_gradient_record():
    raw = """
Processed Input file:
 coordinates geometry.xyz
 run gradient
 cistarget 2
---------------------
Gradient units are Hartree/Bohr
--------------------------------------------------
          dE/dX          dE/dY          dE/dZ
      0.100000      -0.200000       0.300000
     -0.400000       0.500000      -0.600000
--------------------------------------------------
"""
    parsed = parse_terachem_output(raw, raw_str_in=True)

    records = parsed["excited_state_records"]
    assert len(records) == 1
    assert records[0]["source"] == "parse_gradient_section"
    assert records[0]["section_idx"] == 0
    assert records[0]["state_i"] == 0
    assert records[0]["state_j"] == 2
    np.testing.assert_allclose(
        records[0]["energy_gradient"],
        [[0.1, -0.2, 0.3], [-0.4, 0.5, -0.6]],
    )
    assert records[0]["nonadiabatic_coupling"] is None


def test_parse_coupling_run_adds_nonadiabatic_coupling_record():
    raw = """
Processed Input file:
 coordinates geometry.xyz
 run coupling
 nacstate1 1
 nacstate2 3
---------------------
Gradient units are Hartree/Bohr
--------------------------------------------------
          dE/dX          dE/dY          dE/dZ
      0.010000       0.020000       0.030000
     -0.010000      -0.020000      -0.030000
--------------------------------------------------
"""
    parsed = parse_terachem_output(raw, raw_str_in=True)

    records = parsed["excited_state_records"]
    assert len(records) == 1
    assert records[0]["source"] == "parse_gradient_section"
    assert records[0]["section_idx"] == 0
    assert records[0]["state_i"] == 1
    assert records[0]["state_j"] == 3
    assert records[0]["energy_gradient"] is None
    np.testing.assert_allclose(
        records[0]["nonadiabatic_coupling"],
        [[0.01, 0.02, 0.03], [-0.01, -0.02, -0.03]],
    )


def test_parse_casscf_orbitals_without_occupations(tmp_path):
    """Orbital sections without occupation column should parse with nan occupancies."""
    output_path = tmp_path / "terachem_orbitals_no_occ.out"
    output_path.write_text(
        "  Orbital      Energy\n  -------------------\n1 -20.123\n2 -10.456\n\n"
    )
    parsed = parse_terachem_output(str(output_path))

    assert parsed["casscf_orbitals"][1]["orb_energy"][0] == -20.123
    assert parsed["casscf_orbitals"][2]["orb_energy"][0] == -10.456
    assert math.isnan(parsed["casscf_orbitals"][1]["orb_occ"][0])
    assert math.isnan(parsed["casscf_orbitals"][2]["orb_occ"][0])


def test_parse_mixed_spin_casscf_transition_dipoles():
    """Mixed singlet/triplet transition sections should be grouped by multiplicity."""
    raw = """
Singlet state electronic transitions:

 Transition      Tx        Ty        Tz       |T|    Osc. (a.u.)
-----------------------------------------------------------------
   1 ->  2     0.0608   -0.0984   -0.9239    0.9311    0.0641
   1 ->  3     1.5069    1.3524    2.8763    3.5175    0.9676
   1 ->  4    -0.0858   -0.0738   -0.0082    0.1135    0.0013
   2 ->  3     0.4098    0.2017   -0.5490    0.7142    0.0022
   2 ->  4     0.0133   -0.0206   -0.1075    0.1103    0.0004
   3 ->  4    -0.0182   -0.0304    0.0400    0.0534    0.0001

Triplet state electronic transitions:

 Transition      Tx        Ty        Tz       |T|    Osc. (a.u.)
-----------------------------------------------------------------
   1 ->  2     0.3744    0.2630   -0.0233    0.4581    0.0039
   1 ->  3     0.0133   -0.0119   -0.1497    0.1508    0.0013
   2 ->  3     0.1273    0.0638   -0.0064    0.1425    0.0008
"""
    parsed = parse_terachem_output(raw, raw_str_in=True)

    assert "casscf_transition_dipoles" in parsed
    assert "singlet" in parsed["casscf_transition_dipoles"]
    assert "triplet" in parsed["casscf_transition_dipoles"]

    assert parsed["casscf_transition_dipoles"]["singlet"]["1 -> 4"]["osc_strength"] == [
        0.0013
    ]
    assert parsed["casscf_transition_dipoles"]["triplet"]["1 -> 2"]["osc_strength"] == [
        0.0039
    ]


def test_parse_state_resolved_dipole_moments():
    """Singlet and triplet permanent dipoles should be grouped by root."""
    raw = """
Singlet state dipole moments:

  Root       Dx         Dy         Dz         |D|  (Debye)
------------------------------------------------------------
     1    -1.7867    -0.0581     0.0072     1.7877
     2    -1.2158    -0.0204     0.0130     1.2160

Triplet state dipole moments:

  Root       Dx         Dy         Dz         |D|  (Debye)
------------------------------------------------------------
     1    -0.9506    -0.0208     0.0139     0.9509
     2    -1.4492    -0.0575     0.0103     1.4504

Singlet state dipole moments:

  Root       Dx         Dy         Dz         |D|  (Debye)
------------------------------------------------------------
     1    -1.7000    -0.0500     0.0100     1.7010
     2    -1.2000    -0.0200     0.0100     1.2002
"""
    parsed = parse_terachem_output(raw, raw_str_in=True)

    dipoles = parsed["casscf_state_dipoles"]
    assert set(dipoles) == {"singlet", "triplet"}
    assert dipoles["singlet"][1]["Dx"] == [-1.7867, -1.7000]
    assert dipoles["singlet"][1]["Dy"] == [-0.0581, -0.0500]
    assert dipoles["singlet"][1]["Dz"] == [0.0072, 0.0100]
    assert dipoles["singlet"][1]["D_mag"] == [1.7877, 1.7010]
    assert dipoles["singlet"][2]["Dx"] == [-1.2158, -1.2000]
    assert dipoles["triplet"][1]["Dx"] == [-0.9506]
    assert dipoles["triplet"][1]["D_mag"] == [0.9509]
    assert dipoles["triplet"][2]["Dz"] == [0.0103]
    assert dipoles["triplet"][2]["D_mag"] == [1.4504]

    records = [
        record
        for record in parsed["excited_state_records"]
        if record["source"] == "parse_casscf_state_dipole_section"
    ]
    assert len(records) == 6
    assert records[0]["multiplicity"] == "singlet"
    assert records[0]["section_idx"] == 0
    assert records[0]["state_i"] == 0
    assert records[0]["state_j"] == 0
    assert records[0]["Dx"] == -1.7867
    assert records[0]["D_mag"] == 1.7877
    assert records[2]["multiplicity"] == "triplet"
    assert records[2]["section_idx"] == 0
    assert records[2]["state_j"] == 0
    assert records[2]["Dz"] == 0.0139
    assert records[4]["section_idx"] == 1
    assert records[4]["state_j"] == 0
    assert math.isnan(records[0]["Tx"])


def test_parse_casscf_multipole_and_velocity_sections():
    """Velocity, magnetic, and Cartesian quadrupole tables should parse by spin."""
    raw = """
Singlet state velocity transition dipole moments:

 Transition      Px        Py        Pz       |P|    Osc. (a.u.)
-----------------------------------------------------------------
   1 ->  2    -0.0000    0.0002   -0.0005    0.0005    0.0000

Singlet state magnetic transition dipole moments and rotational strengths:

 Transition      Lx        Ly        Lz        |L|    R(vel) (a.u.)
--------------------------------------------------------------------
   1 ->  2     1.0305   -0.0243   -0.0012    1.0308   -0.0005

Triplet state velocity transition dipole moments:

 Transition      Px        Py        Pz       |P|    Osc. (a.u.)
-----------------------------------------------------------------
   1 ->  2    -0.0000    0.0000   -0.0008    0.0008    0.0000

Triplet state magnetic transition dipole moments and rotational strengths:

 Transition      Lx        Ly        Lz        |L|    R(vel) (a.u.)
--------------------------------------------------------------------
   1 ->  2    -0.8823    0.0134   -0.0011    0.8824    0.0001

Singlet state electric quadrupole moments (Cartesian):
Operator origin (a.u.):     2.076016,    -0.000000,    -0.000000

 Root     Qxy        Qxz        Qyz        Qxx        Qyy        Qzz       R^2
--------------------------------------------------------------------------------------------
    1    -0.0480    -0.0079     0.0004   -12.6344   -15.3563   -14.6572   -42.6480
    2    -0.0584    -0.0173    -0.0004   -13.4980   -13.2566   -16.5223   -43.2769

Triplet state electric quadrupole moments (Cartesian):
Operator origin (a.u.):     2.076016,    -0.000000,    -0.000000

 Root     Qxy        Qxz        Qyz        Qxx        Qyy        Qzz       R^2
--------------------------------------------------------------------------------------------
    1    -0.0586    -0.0181    -0.0012   -13.5443   -13.2489   -16.4991   -43.2923
    2    -0.0463    -0.0148     0.0013   -14.1220   -15.3107   -14.5139   -43.9465

Singlet state electric transition quadrupole moments (Cartesian):

 Transition     Qxy        Qxz        Qyz        Qxx        Qyy        Qzz       norm (a.u.)
--------------------------------------------------------------------------------------------
   1 ->  2     0.0030    -0.0100    -0.9855    -0.0015    -0.0008     0.0005     0.9855

Triplet state electric transition quadrupole moments (Cartesian):

 Transition     Qxy        Qxz        Qyz        Qxx        Qyy        Qzz       norm (a.u.)
--------------------------------------------------------------------------------------------
   1 ->  2     0.0022    -0.0175    -0.9056    -0.0004    -0.0014     0.0013     0.9057
"""
    parsed = parse_terachem_output(raw, raw_str_in=True)

    velocity = parsed["casscf_velocity_transition_dipoles"]
    assert velocity["singlet"]["1 -> 2"]["Px"] == [0.0]
    assert velocity["singlet"]["1 -> 2"]["Py"] == [0.0002]
    assert velocity["singlet"]["1 -> 2"]["Pz"] == [-0.0005]
    assert velocity["singlet"]["1 -> 2"]["P_mag"] == [0.0005]
    assert velocity["singlet"]["1 -> 2"]["velocity_osc_strength"] == [0.0]
    assert velocity["triplet"]["1 -> 2"]["Pz"] == [-0.0008]
    assert velocity["triplet"]["1 -> 2"]["P_mag"] == [0.0008]

    magnetic = parsed["casscf_magnetic_transition_dipoles"]
    assert magnetic["singlet"]["1 -> 2"]["Lx"] == [1.0305]
    assert magnetic["singlet"]["1 -> 2"]["Ly"] == [-0.0243]
    assert magnetic["singlet"]["1 -> 2"]["Lz"] == [-0.0012]
    assert magnetic["singlet"]["1 -> 2"]["L_mag"] == [1.0308]
    assert magnetic["singlet"]["1 -> 2"]["rot_strength"] == [-0.0005]
    assert magnetic["triplet"]["1 -> 2"]["Lx"] == [-0.8823]
    assert magnetic["triplet"]["1 -> 2"]["rot_strength"] == [0.0001]

    quadrupoles = parsed["casscf_state_quadrupoles"]
    singlet_root = quadrupoles["singlet"][1]
    assert singlet_root["Qxy"] == [-0.0480]
    assert singlet_root["Qxz"] == [-0.0079]
    assert singlet_root["Qyz"] == [0.0004]
    assert singlet_root["Qxx"] == [-12.6344]
    assert singlet_root["Qyy"] == [-15.3563]
    assert singlet_root["Qzz"] == [-14.6572]
    assert singlet_root["R2"] == [-42.6480]
    assert singlet_root["origin_x"] == [2.076016]
    assert singlet_root["origin_y"] == [0.0]
    assert singlet_root["origin_z"] == [0.0]
    assert quadrupoles["singlet"][2]["Qzz"] == [-16.5223]
    assert quadrupoles["triplet"][1]["Qxx"] == [-13.5443]
    assert quadrupoles["triplet"][2]["R2"] == [-43.9465]

    transition_quadrupoles = parsed["casscf_transition_quadrupoles"]
    singlet_transition = transition_quadrupoles["singlet"]["1 -> 2"]
    assert singlet_transition["Qxy"] == [0.0030]
    assert singlet_transition["Qxz"] == [-0.0100]
    assert singlet_transition["Qyz"] == [-0.9855]
    assert singlet_transition["Qxx"] == [-0.0015]
    assert singlet_transition["Qyy"] == [-0.0008]
    assert singlet_transition["Qzz"] == [0.0005]
    assert singlet_transition["Q_norm"] == [0.9855]
    assert transition_quadrupoles["triplet"]["1 -> 2"]["Qyz"] == [-0.9056]
    assert transition_quadrupoles["triplet"]["1 -> 2"]["Q_norm"] == [0.9057]

    def records_from(source):
        return [
            record
            for record in parsed["excited_state_records"]
            if record["source"] == source
        ]

    velocity_records = records_from("parse_casscf_velocity_transition_dipole_section")
    magnetic_records = records_from("parse_casscf_magnetic_transition_dipole_section")
    quadrupole_records = records_from("parse_casscf_state_quadrupole_section")
    transition_records = records_from("parse_casscf_transition_quadrupole_section")

    assert len(velocity_records) == 2
    assert velocity_records[0]["multiplicity"] == "singlet"
    assert velocity_records[0]["section_idx"] == 0
    assert velocity_records[0]["state_i"] == 0
    assert velocity_records[0]["state_j"] == 1
    assert velocity_records[0]["Py"] == 0.0002
    assert math.isnan(velocity_records[0]["osc_strength"])
    assert velocity_records[1]["multiplicity"] == "triplet"
    assert velocity_records[1]["section_idx"] == 0

    assert magnetic_records[0]["Lx"] == 1.0305
    assert magnetic_records[0]["rot_strength"] == -0.0005
    assert magnetic_records[0]["state_j"] == 1

    assert quadrupole_records[0]["state_i"] == 0
    assert quadrupole_records[0]["state_j"] == 0
    assert quadrupole_records[0]["Qxx"] == -12.6344
    assert quadrupole_records[0]["origin_x"] == 2.076016
    assert quadrupole_records[1]["state_j"] == 1
    assert quadrupole_records[2]["multiplicity"] == "triplet"
    assert quadrupole_records[2]["section_idx"] == 0

    assert transition_records[0]["Qyz"] == -0.9855
    assert transition_records[0]["Q_norm"] == 0.9855
    assert transition_records[0]["state_i"] == 0
    assert transition_records[0]["state_j"] == 1
    assert transition_records[1]["multiplicity"] == "triplet"
    assert transition_records[1]["section_idx"] == 0


def test_parse_fms_coordinates():
    """FMS geometries should stack as Angstrom frames without consuming later sections."""
    raw = """
Receiving coordinates from FMS
C        -0.03219779        0.00172618       -0.00127161
S         1.63336114       -0.00201617        0.00012594
H        -0.72094791        0.95766053        0.00557171
H        -0.58319249       -0.91425953        0.00557171

FINAL ENERGY: -400.123 a.u.

Receiving coordinates from FMS
C        -0.10000000        0.00000000        0.00000000
S         1.60000000        0.00000000        0.00000000
H        -0.70000000        0.90000000        0.00000000
H        -0.50000000       -0.90000000        0.00000000
"""
    parsed = parse_terachem_output(raw, raw_str_in=True)

    coordinates = parsed["fms_coordinates"]
    assert coordinates["elements"] == ["C", "S", "H", "H"]
    assert coordinates["units"] == "angstrom"
    assert coordinates["xyz"].shape == (2, 4, 3)
    np.testing.assert_allclose(
        coordinates["xyz"][0, 0],
        [-0.03219779, 0.00172618, -0.00127161],
    )
    np.testing.assert_allclose(
        coordinates["xyz"][0, 1],
        [1.63336114, -0.00201617, 0.00012594],
    )
    np.testing.assert_allclose(
        coordinates["xyz"][1, 2],
        [-0.70000000, 0.90000000, 0.00000000],
    )
    assert parsed["scf_energy"] == [-400.123]


def test_parse_fms_coordinates_rejects_changed_element_order():
    raw = """
Receiving coordinates from FMS
C 0.0 0.0 0.0
H 1.0 0.0 0.0

Receiving coordinates from FMS
H 0.0 0.0 0.0
C 1.0 0.0 0.0
"""
    with pytest.raises(ValueError, match="Element order changed"):
        parse_terachem_output(raw, raw_str_in=True)


def test_parse_spin_orbit_coupling_matrices():
    """SOC headers name the states, and each block keeps its (S, Sz) matrix."""
    raw = """
Triplet[0] - Triplet[0] Spin-orbit coupling (cm-1):

   (S, Sz)        (1.0, 1.0)           (1.0, 0.0)           (1.0,-1.0)
--------------------------------------------------------------------------------
 (1.0, 1.0)    0.0000 +   0.0000i     0.0000 +   0.0000i     0.0000 +   0.0000i
 (1.0, 0.0)   -0.0000 +   0.0000i     0.0000 +   0.0000i     0.0000 +   0.0000i
 (1.0,-1.0)    0.0000 +   0.0000i    -0.0000 +   0.0000i     0.0000 -   0.0000i

Triplet[0] - Triplet[1] Spin-orbit coupling (cm-1):

   (S, Sz)        (1.0, 1.0)           (1.0, 0.0)           (1.0,-1.0)
--------------------------------------------------------------------------------
 (1.0, 1.0)    0.0000 -   0.2513i     2.7643 - 148.1166i     0.0000 +   0.0000i
 (1.0, 0.0)   -2.7643 - 148.1166i     0.0000 +  -0.0000i     2.7643 - 148.1166i
 (1.0,-1.0)    0.0000 +   0.0000i    -2.7643 - 148.1166i     0.0000 +   0.2513i

Triplet[1] - Triplet[1] Spin-orbit coupling (cm-1):

   (S, Sz)        (1.0, 1.0)           (1.0, 0.0)           (1.0,-1.0)
--------------------------------------------------------------------------------
 (1.0, 1.0)    0.0000 -   0.0000i    -0.0000 +   0.0000i     0.0000 +   0.0000i
 (1.0, 0.0)    0.0000 +   0.0000i     0.0000 +  -0.0000i    -0.0000 +   0.0000i
 (1.0,-1.0)    0.0000 +   0.0000i     0.0000 +   0.0000i     0.0000 +   0.0000i

Singlet[0] - Triplet[0] Spin-orbit coupling (cm-1):

   (S, Sz)        (1.0, 1.0)           (1.0, 0.0)           (1.0,-1.0)
--------------------------------------------------------------------------------
 (0.0, 0.0)    3.0733 + 156.8173i     0.0000 +   0.3739i     3.0733 - 156.8173i

Singlet[0] - Triplet[1] Spin-orbit coupling (cm-1):

   (S, Sz)        (1.0, 1.0)           (1.0, 0.0)           (1.0,-1.0)
--------------------------------------------------------------------------------
 (0.0, 0.0)    0.3748 -   0.0725i     0.0000 -   0.0027i     0.3748 +   0.0725i

Singlet[1] - Triplet[0] Spin-orbit coupling (cm-1):

   (S, Sz)        (1.0, 1.0)           (1.0, 0.0)           (1.0,-1.0)
--------------------------------------------------------------------------------
 (0.0, 0.0)   -0.0201 -   0.0642i     0.0000 +   0.0002i    -0.0201 +   0.0642i

Singlet[1] - Triplet[1] Spin-orbit coupling (cm-1):

   (S, Sz)        (1.0, 1.0)           (1.0, 0.0)           (1.0,-1.0)
--------------------------------------------------------------------------------
 (0.0, 0.0)   -2.8248 - 151.0035i     0.0000 +   0.2277i    -2.8248 + 151.0035i

FINAL ENERGY: -400.0 a.u.
"""
    parsed = parse_terachem_output(raw, raw_str_in=True)

    soc = parsed["spin_orbit_couplings"]
    assert soc["units"] == "cm-1"
    triplet = soc["couplings"]["triplet -> triplet"]
    assert set(triplet) == {"0 -> 0", "0 -> 1", "1 -> 1"}
    pair = triplet["0 -> 1"]
    assert pair["state_i"] == 0
    assert pair["state_j"] == 1
    assert pair["multiplicity_i"] == "triplet"
    assert pair["multiplicity_j"] == "triplet"
    np.testing.assert_allclose(pair["row_S"], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(pair["row_Sz"], [1.0, 0.0, -1.0])
    np.testing.assert_allclose(pair["col_S"], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(pair["col_Sz"], [1.0, 0.0, -1.0])
    assert pair["soc"].shape == (1, 3, 3)
    np.testing.assert_allclose(
        pair["soc"][0],
        [
            [0.0 - 0.2513j, 2.7643 - 148.1166j, 0.0],
            [-2.7643 - 148.1166j, 0.0, 2.7643 - 148.1166j],
            [0.0, -2.7643 - 148.1166j, 0.0 + 0.2513j],
        ],
    )
    assert np.signbit(pair["soc"][0, 1, 1].imag)

    singlet = soc["couplings"]["singlet -> triplet"]["0 -> 0"]
    assert singlet["state_i"] == 0
    assert singlet["state_j"] == 0
    np.testing.assert_allclose(singlet["row_S"], [0.0])
    np.testing.assert_allclose(singlet["row_Sz"], [0.0])
    assert singlet["soc"].shape == (1, 1, 3)
    np.testing.assert_allclose(
        singlet["soc"][0, 0],
        [3.0733 + 156.8173j, 0.0 + 0.3739j, 3.0733 - 156.8173j],
    )
    excited = soc["couplings"]["singlet -> triplet"]["1 -> 1"]
    assert excited["state_i"] == 1
    assert excited["state_j"] == 1
    np.testing.assert_allclose(
        excited["soc"][0, 0],
        [-2.8248 - 151.0035j, 0.0 + 0.2277j, -2.8248 + 151.0035j],
    )
    assert parsed["scf_energy"] == [-400.0]

    records = [
        record
        for record in parsed["excited_state_records"]
        if record["source"] == "parse_soc_section"
    ]
    assert len(records) == 7
    assert records[1]["state_i"] == 0
    assert records[1]["state_j"] == 1
    assert records[1]["multiplicity_i"] == "triplet"
    assert records[1]["multiplicity_j"] == "triplet"
    assert records[1]["section_idx"] == 0
    assert records[1]["soc_units"] == "cm-1"
    np.testing.assert_allclose(records[1]["soc"], pair["soc"][0])
    assert records[3]["multiplicity_i"] == "singlet"
    assert records[3]["section_idx"] == 0
    assert records[3]["soc"].shape == (1, 3)


def test_parse_spin_orbit_coupling_stacks_repeated_pairs():
    raw = """
Singlet[0] - Triplet[1] Spin-orbit coupling (cm-1):

   (S, Sz)        (1.0, 1.0)           (1.0, 0.0)           (1.0,-1.0)
--------------------------------------------------------------------------------
 (0.0, 0.0)    0.1000 +   0.2000i     0.3000 +   0.4000i     0.5000 +   0.6000i

Singlet[0] - Triplet[1] Spin-orbit coupling (cm-1):

   (S, Sz)        (1.0, 1.0)           (1.0, 0.0)           (1.0,-1.0)
--------------------------------------------------------------------------------
 (0.0, 0.0)    1.1000 +   1.2000i     1.3000 +   1.4000i     1.5000 +   1.6000i
"""
    parsed = parse_terachem_output(raw, raw_str_in=True)
    coupling = parsed["spin_orbit_couplings"]["couplings"]["singlet -> triplet"][
        "0 -> 1"
    ]
    assert coupling["soc"].shape == (2, 1, 3)
    assert coupling["soc"][0, 0, 0] == pytest.approx(0.1 + 0.2j)
    assert coupling["soc"][1, 0, 2] == pytest.approx(1.5 + 1.6j)
    records = [
        record
        for record in parsed["excited_state_records"]
        if record["source"] == "parse_soc_section"
    ]
    assert [record["section_idx"] for record in records] == [0, 1]


def test_parse_eomccsd_energies_and_transition_properties():
    """EOM-CCSD energy and transition-properties sections should parse."""
    raw = """
  --------------------------------------------------------------------
  ========================> EOM-CCSD Energies <=======================
  --------------------------------------------------------------------
  Root      Total Energy (au)      Ex. Energy (au)     Ex. Energy (eV)
  --------------------------------------------------------------------
    0    -1404.4557280091889879
    1    -1404.3385943668226901       0.1171336424       3.1873680532
    2    -1404.3116931618417311       0.1440348473       3.9193869644
    3    -1404.3018879358712638       0.1538400733       4.1862006943
  --------------------------------------------------------------------

  --------------------------------------------------------------------
  =================> EOM-CCSD Transition Properties <=================
  --------------------------------------------------------------------
  Transition       Excitation Energy (eV)       Oscillator Strength (au)
  --------------------------------------------------------------------
    0 ->   1               3.18739835                     0.88155009
    0 ->   2               3.91941497                     0.02048534
    0 ->   3               4.18621136                     0.06915175
    1 ->   2               0.73201662                     0.02402609
    1 ->   3               0.99881301                     0.02457466
    2 ->   3               0.26679639                     0.00351124
  --------------------------------------------------------------------
"""
    parsed = parse_terachem_output(raw, raw_str_in=True)

    assert "eomccsd_energies" in parsed
    energies = parsed["eomccsd_energies"][0]
    assert energies[0]["total_energy_au"] == -1404.455728009189
    assert math.isnan(energies[0]["exc_energy_au"])
    assert math.isnan(energies[0]["exc_energy_ev"])
    assert energies[3]["exc_energy_ev"] == 4.1862006943

    assert "eomccsd_transition" in parsed
    transitions = parsed["eomccsd_transition"][0]
    assert transitions["0 -> 1"]["exc_energy_ev"] == 3.18739835
    assert transitions["0 -> 1"]["osc_strength"] == 0.88155009
    assert transitions["2 -> 3"]["exc_energy_ev"] == 0.26679639

    state_records = [
        r
        for r in parsed["excited_state_records"]
        if r["source"] == "parse_eomccsd_energies"
    ]
    transition_records = [
        r
        for r in parsed["excited_state_records"]
        if r["source"] == "parse_eomccsd_transition_properties"
    ]
    assert len(state_records) == 4
    assert len(transition_records) == 6
    assert transition_records[0]["state_i"] == 0
    assert transition_records[0]["state_j"] == 1


def test_parse_eomccsd_transition_mu_elements():
    """Headerless <i|mu|j>: x y z lines should parse as transition dipoles."""
    raw = """
<0|mu|1>:     2.219085     1.672374     1.575919
<1|mu|0>:     2.454885     1.850418     1.742941

<0|mu|2>:    -0.303579    -0.312271    -0.009126
<2|mu|0>:    -0.333022    -0.359734     0.010645

<0|mu|3>:    -0.297247    -0.193969     0.694646
<3|mu|0>:    -0.320314    -0.231101     0.769048

<1|mu|2>:    -0.774388    -0.657218    -0.477613
<2|mu|1>:    -0.823757    -0.693573    -0.514963

<1|mu|3>:     0.717595     0.530096     0.460777
<3|mu|1>:     0.715936     0.526148     0.459217

<2|mu|3>:     0.189998     0.089503    -0.727510
<3|mu|2>:     0.155043     0.077870    -0.688314
"""
    parsed = parse_terachem_output(raw, raw_str_in=True)

    assert "eomccsd_transition_dipoles" in parsed
    transitions = parsed["eomccsd_transition_dipoles"][0]
    assert transitions["0 -> 1"]["Tx"] == 2.219085
    assert transitions["0 -> 1"]["Ty"] == 1.672374
    assert transitions["0 -> 1"]["Tz"] == 1.575919
    assert transitions["0 -> 1"]["T_mag"] == pytest.approx(
        math.sqrt(2.219085**2 + 1.672374**2 + 1.575919**2)
    )
    assert math.isnan(transitions["0 -> 1"]["osc_strength"])

    assert transitions["1 -> 0"]["Tx"] == 2.454885
    assert transitions["1 -> 0"]["Ty"] == 1.850418
    assert transitions["1 -> 0"]["Tz"] == 1.742941
    assert transitions["2 -> 3"]["Tz"] == -0.727510
    assert transitions["3 -> 2"]["Tz"] == -0.688314

    mu_records = [
        r
        for r in parsed["excited_state_records"]
        if r["source"] == "parse_eomccsd_transition_mu_elements"
    ]
    assert len(mu_records) == 12
    assert mu_records[0]["state_i"] == 0
    assert mu_records[0]["state_j"] == 1
    assert mu_records[0]["Tx"] == 2.219085


def test_parse_eomccsd_transition_properties_rejects_five_values():
    """Five-value rows in EOM-CCSD transition properties should raise."""
    raw = """
  --------------------------------------------------------------------
  =================> EOM-CCSD Transition Properties <=================
  --------------------------------------------------------------------
  Transition       Excitation Energy (eV)       Oscillator Strength (au)
  --------------------------------------------------------------------
    0 ->   1         1.0000000      2.0000000      3.0000000      4.0000000      5.0000000
  --------------------------------------------------------------------
"""
    with pytest.raises(
        ValueError,
        match="EOM-CCSD transition properties rows with five values are invalid",
    ):
        parse_terachem_output(raw, raw_str_in=True)


def test_args_importable():
    """Test that args modules can be imported without error."""
    from molzen.io.terachem.args.cas import CAS_FON_ENERGY, CAS_GRAD, CAS_KWARGS
    from molzen.io.terachem.args.cc import CCSD_ARGS, EOMCCSD_ARGS
    from molzen.io.terachem.args.dft import HH_TDA, WPBE, hhtda_fon
    from molzen.io.terachem.args.hf import FON_KWARGS, HF_ENERGY

    assert all(
        item is not None
        for item in (
            CAS_FON_ENERGY,
            CAS_GRAD,
            CAS_KWARGS,
            CCSD_ARGS,
            EOMCCSD_ARGS,
            HH_TDA,
            WPBE,
            hhtda_fon,
            FON_KWARGS,
            HF_ENERGY,
        )
    )
