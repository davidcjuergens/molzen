"""Test terachem output parsing"""

import math
import os

import pytest

from molzen.io.terachem.parse import parse_terachem_output

pytestmark = pytest.mark.local_only

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


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

    assert parsed["excited_states"][es_entry - 1][root]["abs_energy"] == -248.27323993
    assert parsed["excited_states"][es_entry - 1][root]["exc_energy"] == 4.04505765
    assert parsed["excited_states"][es_entry - 1][root]["osc_strength"] == 0.0030
    assert parsed["excited_states"][es_entry - 1][root]["s_squared"] == 0.0000
    assert parsed["excited_states"][es_entry - 1][root]["max_ci_coeff"] == -0.737191

    # should be 5 total entries
    assert len(parsed["excited_states"]) == 5

    # should be 10 excited states parsed for this entry
    assert len(parsed["excited_states"][es_entry - 1]) == 10

    root = 10
    assert parsed["excited_states"][es_entry - 1][root]["abs_energy"] == -248.05883516
    assert parsed["excited_states"][es_entry - 1][root]["exc_energy"] == 9.87930744
    assert parsed["excited_states"][es_entry - 1][root]["osc_strength"] == 0.0014
    assert parsed["excited_states"][es_entry - 1][root]["s_squared"] == 0.0000
    assert parsed["excited_states"][es_entry - 1][root]["max_ci_coeff"] == 0.639490


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
