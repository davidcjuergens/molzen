"""Parsing terachem outputs"""

import re
from typing import Union
import numpy as np


def parse_terachem_output(
    file_path: str,
    custom_section_parsers: Union[None, dict] = None,
    raw_str_in: bool = False,
) -> dict:
    """
    Parse a terachem output file and return relevant data.
    Args:
        file_path (str): Path to the terachem output file.
        custom_section_parsers (dict, optional): Additional section parsers to include for custom TC outputs.
            Dict keys should be strings to search for in a line to identify the section start.
            Dict values should be (name, callable) tuples, where name is the key to store data under,
            and callable is a function that takes (lines, start_idx) and returns (section_dict, end_idx).
        raw_str_in (bool, optional): If True, file_path is treated as raw string input rather than a file path.
    """
    #### Some constants ####
    INPUT_ARGS_HEADER = "Processed Input file:"
    GROUND_STATE_ENERGY_HEADER = "FINAL ENERGY:"
    EXCITED_STATES_RESULTS_HEADER = "Final Excited State Results"

    CASSCF_EXCITED_STATES_HEADER = (
        "Root   Mult.   Total Energy (a.u.)   Ex. Energy (a.u.)     Ex. Energy (eV)"
    )
    CASSCF_SINGLET_TRANSITION_DIPOLE_HEADER = "Singlet state electronic transitions:"
    CASSCF_TRIPLET_TRANSITION_DIPOLE_HEADER = "Triplet state electronic transitions:"
    CASSCF_ORB_ENERGIES_HEADER = "Orbital      Energy"

    #### End constants ####
    if not raw_str_in:
        with open(file_path, "r") as tc_file:
            lines = tc_file.readlines()
    else:
        lines = file_path.splitlines()

    out = {}
    i = 0
    while i < len(lines):
        line = lines[i]

        ### Input Arguments ###
        if INPUT_ARGS_HEADER in line:
            assert not out.get("input_args"), "Input args already parsed!"

            tc_kwargs, i = parse_tc_input_flags(lines, start=i)

            out["input_args"] = tc_kwargs
            continue

        ### Ground State Results ###
        if GROUND_STATE_ENERGY_HEADER in line:
            energy_line = line.strip().split()
            ground_state_energy = float(energy_line[2])

            if out.get("ground_state_energy", None) is None:
                out["ground_state_energy"] = [ground_state_energy]
            else:
                out["ground_state_energy"].append(ground_state_energy)

            i += 1
            continue

        #### Excited State Results ###
        if EXCITED_STATES_RESULTS_HEADER in line:
            excited_state_data, i = parse_excited_state_section(lines, start=i)

            # account for possibility of multiple excited state sections
            if out.get("excited_states", None) is None:
                out["excited_states"] = [excited_state_data]
            else:
                out["excited_states"].append(excited_state_data)
            continue

        ### CASSCF-like excited state results section ###
        if CASSCF_EXCITED_STATES_HEADER in line:
            excited_state_data, i = parse_casscf_excited_state_section(lines, start=i)
            if out.get("casscf_energies", None) is None:
                out["casscf_energies"] = [excited_state_data]
            else:
                out["casscf_energies"].append(excited_state_data)
            continue

        ### CASSCF-like transition dipole results section ###
        transition_multiplicity = None
        if CASSCF_SINGLET_TRANSITION_DIPOLE_HEADER in line:
            transition_multiplicity = "singlet"
        elif CASSCF_TRIPLET_TRANSITION_DIPOLE_HEADER in line:
            transition_multiplicity = "triplet"

        if transition_multiplicity is not None:
            transition_dipole_data, i = parse_casscf_transition_dipole_section(
                lines, start=i, multiplicity=transition_multiplicity
            )
            if out.get("casscf_transition_dipoles", None) is None:
                out["casscf_transition_dipoles"] = [transition_dipole_data]
            else:
                out["casscf_transition_dipoles"].append(transition_dipole_data)
            continue

        ### CASSCF-like orbital energies and occupations ###
        if CASSCF_ORB_ENERGIES_HEADER in line:
            orb_occ_data, i = parse_casscf_orbitals(lines, start=i)
            if out.get("casscf_orbitals", None) is None:
                out["casscf_orbitals"] = [orb_occ_data]
            else:
                out["casscf_orbitals"].append(orb_occ_data)
            continue

        # CUSTOM SECTION PARSERS
        if custom_section_parsers is not None:
            parsed_custom = False
            for section_start, (name, section_parser) in custom_section_parsers.items():
                if section_start in line:
                    start = i  # we have to start at i and let the function handle it
                    section_dict, i = section_parser(lines, start=start)
                    assert not out.get(name), f"{name} already parsed!"
                    out[name] = section_dict
                    parsed_custom = True
                    break
            if parsed_custom:
                continue

        i += 1

    # Post-processing
    if out.get("casscf_energies") is not None:
        postprocess_casscf_energies(out)
    if out.get("casscf_transition_dipoles") is not None:
        postprocess_casscf_transition_dipoles(out)
    if out.get("casscf_orbitals") is not None:
        postprocess_casscf_orbitals(out)
    return out


def parse_casscf_orbitals(lines: list, start: int) -> dict:
    """Parse a casscf-like orbital energy and occupation section"""

    def check_end(myline):
        return myline.isspace()

    def orb_occ_line_parser(myline):
        parts = myline.strip().split()
        assert len(parts) in (2, 3)
        orb_num = int(parts[0])
        orb_energy = float(parts[1])
        orb_occ = float(parts[2]) if len(parts) == 3 else float("nan")

        out = dict(
            orb_num=orb_num,
            orb_energy=orb_energy,
            orb_occ=orb_occ,
        )

        return out

    j = start + 2  # skip header lines
    out = {}
    while not check_end(lines[j]):
        orb_occ_data = orb_occ_line_parser(lines[j])
        orb_num = orb_occ_data.pop("orb_num")
        out[orb_num] = orb_occ_data
        j += 1
    return out, j


def postprocess_casscf_orbitals(out):
    """Organize the CASSCF orbital energy and occupation data by orbital number and put into more convenient format."""
    match_orbs = None
    match_orb_keys = None

    orbs = out["casscf_orbitals"][0].keys()
    example_orb = list(orbs)[0]

    # same orbitals in all sections
    if match_orbs is not None:
        assert set(orbs) == set(match_orbs)
    else:
        match_orbs = orbs

    # same keys for each orbital in all sections
    orb_keys = out["casscf_orbitals"][0][example_orb].keys()
    if match_orb_keys is not None:
        assert set(orb_keys) == set(match_orb_keys)
    else:
        match_orb_keys = orb_keys

    o = {}
    for orb in orbs:
        o[orb] = {k: [] for k in orb_keys}
        for section in out["casscf_orbitals"]:
            for k in orb_keys:
                o[orb][k].append(section[orb][k])

    out["casscf_orbitals"] = o


def postprocess_casscf_transition_dipoles(out):
    """Organize the CASSCF transition dipole data by root and put into more convenient format."""
    def merge_transition_sections(sections):
        """Merge a list of transition-dipole sections into transition-keyed arrays."""
        possible_transitions = []
        transition_keys = []

        for section in sections:
            for transition, values in section.items():
                if transition not in possible_transitions:
                    possible_transitions.append(transition)
                for key in values.keys():
                    if key not in transition_keys:
                        transition_keys.append(key)

        merged = {
            transition: {k: [] for k in transition_keys}
            for transition in possible_transitions
        }
        for section in sections:
            for transition in possible_transitions:
                values = section.get(transition)
                for key in transition_keys:
                    merged[transition][key].append(
                        np.nan if values is None else values.get(key, np.nan)
                    )
        return merged

    # New parser stores sections as {"multiplicity": ..., "transitions": ...}; keep support
    # for legacy shape where each section is just transition -> data.
    grouped_sections = {}
    for section in out["casscf_transition_dipoles"]:
        if (
            isinstance(section, dict)
            and "multiplicity" in section
            and "transitions" in section
        ):
            multiplicity = section["multiplicity"]
            transitions = section["transitions"]
        else:
            multiplicity = None
            transitions = section
        grouped_sections.setdefault(multiplicity, []).append(transitions)

    out["casscf_transition_dipoles"] = {
        multiplicity: merge_transition_sections(sections)
        for multiplicity, sections in grouped_sections.items()
    }


def postprocess_casscf_energies(out):
    """Organize the CASSCF state data by root and put into more convenient format."""
    match_roots = None
    match_root_keys = None

    roots = out["casscf_energies"][0].keys()

    # same roots in all sections
    if match_roots is not None:
        assert set(roots) == set(match_roots)
    else:
        match_roots = roots

    # same keys for each root in all sections
    root_keys = out["casscf_energies"][0][1].keys()
    if match_root_keys is not None:
        assert set(root_keys) == set(match_root_keys)
    else:
        match_root_keys = root_keys

    o = {}
    for root in roots:
        o[root] = {k: [] for k in root_keys}
        for section in out["casscf_energies"]:
            for k in root_keys:
                o[root][k].append(section[root][k])

    out["casscf_energies"] = o


def parse_casscf_transition_dipole_section(
    lines: list, start: int, multiplicity: Union[None, str] = None
) -> dict:
    """Parse a casscf-like transition dipole results section"""

    def transition_dipole_line_parser(myline):
        transition_match = re.match(r"\s*(\d+)\s*->\s*(\d+)\s+(.*)", myline)
        assert transition_match is not None

        transition_state1 = int(transition_match.group(1))
        transition_state2 = int(transition_match.group(2))
        parts = transition_match.group(3).strip().split()
        assert len(parts) >= 5

        transition = f"{transition_state1} -> {transition_state2}"
        Tx = float(parts[0])
        Ty = float(parts[1])
        Tz = float(parts[2])
        T_mag = float(parts[3])
        osc_strength = float(parts[4])

        out = dict(
            transition=transition,
            Tx=Tx,
            Ty=Ty,
            Tz=Tz,
            T_mag=T_mag,
            osc_strength=osc_strength,
        )

        return out

    def is_transition_header_line(myline):
        stripped = myline.strip().lower()
        return stripped.startswith("transition") or set(stripped) == {"-"}

    j = start + 1
    out = {}
    while j < len(lines):
        line = lines[j]
        stripped = line.strip()

        if not stripped:
            if out:
                break
            j += 1
            continue

        if is_transition_header_line(line):
            j += 1
            continue

        try:
            td_data = transition_dipole_line_parser(line)
        except Exception:
            break

        transition = td_data.pop("transition")
        out[transition] = td_data
        j += 1

    if multiplicity is None:
        return out, j
    return {"multiplicity": multiplicity, "transitions": out}, j


def parse_tc_input_flags(lines: list, start: int) -> dict:
    """Parse the input flags to terachem

    Args:
        lines (list): List of lines from the terachem output file.
        start (int): Index of the line where the input flags section starts.
    """
    # ensure we are at the start of section
    assert "Processed Input file:" in lines[start]

    def flag_line_parser(myline):
        return myline.strip().split()

    def check_end(myline):
        return "---------------------" in myline

    j = start + 1  # skip header line
    out = {}

    while not check_end(lines[j]):
        line = lines[j]
        stripped = line.strip()

        if not stripped:
            j += 1
            continue

        if stripped == "$constraints":
            constraints = []
            j += 1
            while j < len(lines):
                constraint_line = lines[j].strip()
                if constraint_line == "$end":
                    break
                if constraint_line:
                    constraints.append(constraint_line)
                j += 1
            out["constraints"] = constraints
            j += 1
            continue

        try:
            key, val = stripped.split(None, 1)
        except Exception as e:
            print(f"Error parsing line {j}: {lines[j]}")
            print(f"Line j+1: {lines[j + 1]}")
            raise e
        out[key] = val
        j += 1

    return out, j


def parse_casscf_excited_state_section(lines: list, start: int) -> dict:
    """Parse a casscf-like excited state results section"""

    def check_end(myline):
        return myline.isspace()

    def excited_state_line_parser(myline):
        parts = myline.strip().split()
        assert len(parts) in (3, 5)  # 3 for root 1, 5 for all others
        root = int(parts[0])
        mult_string = parts[1]  # 'singlet', 'triplet', etc
        total_energy_au = float(parts[2])
        if len(parts) == 5:
            exc_energy_au = float(parts[3])
            exc_energy_ev = float(parts[4])
        else:
            exc_energy_au = float("nan")
            exc_energy_ev = float("nan")

        out = dict(
            root=root,
            multiplicity=mult_string,
            total_energy_au=total_energy_au,
            exc_energy_au=exc_energy_au,
            exc_energy_ev=exc_energy_ev,
        )

        return out

    j = start + 2  # skip header lines
    out = {}
    while not check_end(lines[j]):
        state_data = excited_state_line_parser(lines[j])
        root = state_data.pop("root")
        out[root] = state_data
        j += 1
    return out, j


def parse_excited_state_section(lines: list, start: int) -> dict:
    """Parse an excited state results section from terachem output.

    Args:
        lines (list): List of lines from the terachem output file.
        start (int): Index of the line where the excited state section starts.
    """
    assert "Final Excited State Results" in lines[start]

    def check_end(myline):
        return myline.isspace()

    def excited_state_line_parser(myline):
        parts = myline.strip().split()
        out = {
            "root": int(parts[0]),
            "abs_energy": float(parts[1]),
            "exc_energy": float(parts[2]),
            "osc_strength": float(parts[3]),
            "s_squared": float(parts[4]),
            "max_ci_coeff": float(parts[5]),
        }
        return out

    j = start + 4

    out = {}
    while not check_end(lines[j]):
        state_data = excited_state_line_parser(lines[j])
        root = state_data.pop("root")
        out[root] = state_data
        j += 1
    return out, j


def soc_parser(lines: list[str], start: int, n_S: int = 11, n_T: int = 11) -> dict:
    """Parsers SOC section from terachem output.

    Args:
        lines (list[str]): List of lines from the terachem output file.
        start (int): Index to start parsing from.
        n_S (int): Number of singlet states INCLUDING S0.
        n_T (int): Number of triplet states INCLUDING T0.
    """

    assert "SOC between" in lines[start]

    def check_end(myline):
        return "Total processing time" in myline

    soc_matrix = np.full((n_S, n_T), np.nan)
    j = start
    while not check_end(lines[j]):
        line = lines[j]

        if "SOC between" in line:
            split = line.strip().split()
            singlet_idx = int(split[3][1:])
            triplet_idx = int(split[5][1:])
        elif "SOC Constant" in line:
            SOC_constant = float(line.strip().split()[-1])
            soc_matrix[singlet_idx, triplet_idx] = SOC_constant

        j += 1

    return {"soc_matrix": soc_matrix}, j
