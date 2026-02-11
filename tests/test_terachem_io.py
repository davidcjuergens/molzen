"""Test terachem output parsing"""

import os

import pytest

from molzen.io.terachem.parse import parse_terachem_output

pytestmark = pytest.mark.local_only

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

def test_parse_casscf_optimization():
    """Tests the parsing of a CASSCF optimization output file from terachem."""
    output_path = os.path.join(THIS_DIR, "data/terachem_casscf_s0_opt.out")
    parsed = parse_terachem_output(output_path)

    assert "casscf_states" in parsed
    assert len(parsed["casscf_states"]) == 4
    assert parsed["casscf_states"][1]["total_energy_au"][0] == -1954.15973814
    assert parsed["casscf_states"][2]["total_energy_au"][0] == -1954.01500372
    assert parsed["casscf_states"][3]["total_energy_au"][0] == -1953.98558982
    assert parsed["casscf_states"][4]["total_energy_au"][0] == -1953.97030442

    assert len(parsed["casscf_states"][1]["total_energy_au"]) == 357


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
    
    assert parsed["excited_states"][es_entry-1][root]["abs_energy"] == -248.27323993
    assert parsed["excited_states"][es_entry-1][root]["exc_energy"] == 4.04505765
    assert parsed["excited_states"][es_entry-1][root]["osc_strength"] == 0.0030
    assert parsed["excited_states"][es_entry-1][root]["s_squared"] == 0.0000
    assert parsed["excited_states"][es_entry-1][root]["max_ci_coeff"] == -0.737191

    # should be 5 total entries
    assert len(parsed["excited_states"]) == 5

    # should be 10 excited states parsed for this entry
    assert len(parsed["excited_states"][es_entry-1]) == 10

    root = 10
    assert parsed["excited_states"][es_entry-1][root]["abs_energy"] == -248.05883516
    assert parsed["excited_states"][es_entry-1][root]["exc_energy"] == 9.87930744
    assert parsed["excited_states"][es_entry-1][root]["osc_strength"] == 0.0014
    assert parsed["excited_states"][es_entry-1][root]["s_squared"] == 0.0000
    assert parsed["excited_states"][es_entry-1][root]["max_ci_coeff"] == 0.639490
