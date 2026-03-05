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

    assert "eomccsd_transition_mu" in parsed
    transitions = parsed["eomccsd_transition_mu"][0]
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
