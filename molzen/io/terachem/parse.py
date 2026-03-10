"""Parsing terachem outputs"""

import inspect
import re
from typing import Union
import numpy as np

STANDARDIZED_EXCITED_RECORD_DEFAULTS = {
    "source": None,
    "section_idx": np.nan,
    "state_i": np.nan,
    "state_j": np.nan,
    "multiplicity": None,
    "total_energy_au": np.nan,
    "exc_energy_au": np.nan,
    "exc_energy_ev": np.nan,
    "exc_energy_nm": np.nan,
    "osc_strength": np.nan,
    "Tx": np.nan,
    "Ty": np.nan,
    "Tz": np.nan,
    "T_mag": np.nan,
    "s_squared": np.nan,
    "max_ci_coeff": np.nan,
}


def make_standardized_excited_record(**kwargs) -> dict:
    """Create one standardized excited-state record with defaults for missing fields."""
    unknown_keys = sorted(set(kwargs.keys()) - set(STANDARDIZED_EXCITED_RECORD_DEFAULTS))
    if unknown_keys:
        raise KeyError(
            "Unexpected keys for standardized excited-state record: "
            f"{', '.join(unknown_keys)}"
        )

    record = STANDARDIZED_EXCITED_RECORD_DEFAULTS.copy()
    record.update(kwargs)
    return record


def build_standardized_excited_records(records: list[dict]) -> list[dict]:
    """Normalize row dictionaries to the standardized excited-state record schema."""
    return [make_standardized_excited_record(**record) for record in records]


def is_casscf_like_excited_state_header(myline: str) -> bool:
    """Return True for CASSCF/HH-TDA-like excited-state energy table headers."""
    stripped = myline.strip()
    if not stripped:
        return False

    normalized = " ".join(stripped.lower().split())
    if not normalized.startswith("root"):
        return False

    required_tokens = (
        "mult.",
        "total energy (a.u.)",
        "ex. energy (a.u.)",
        "ex. energy (ev)",
    )
    return all(token in normalized for token in required_tokens)


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
    SCF_ENERGY_HEADER = "FINAL ENERGY:"
    EXCITED_STATES_RESULTS_HEADER = "Final Excited State Results"
    CASSCF_SINGLET_TRANSITION_DIPOLE_HEADER = "Singlet state electronic transitions:"
    CASSCF_TRIPLET_TRANSITION_DIPOLE_HEADER = "Triplet state electronic transitions:"
    CASSCF_ORB_ENERGIES_HEADER = "Orbital      Energy"
    EOMCCSD_ENERGIES_HEADER = "====> EOM-CCSD Energies <===="
    EOMCCSD_TRANSITION_PROPERTIES_HEADER = "====> EOM-CCSD Transition Properties <===="
    #### End constants ####

    if not raw_str_in:
        with open(file_path, "r") as tc_file:
            lines = tc_file.readlines()
    else:
        lines = file_path.splitlines()

    out = {}
    standardized_excited_records = []
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
        if SCF_ENERGY_HEADER in line:
            energy_line = line.strip().split()
            scf_energy = float(energy_line[2])

            if out.get("scf_energy", None) is None:
                out["scf_energy"] = [scf_energy]
            else:
                out["scf_energy"].append(scf_energy)

            i += 1
            continue

        #### Excited State Results ###
        if EXCITED_STATES_RESULTS_HEADER in line:
            section_idx = len(out.get("excited_states", []))
            excited_state_data, section_records, i = parse_excited_state_section(
                lines, start=i, return_standardized_records=True
            )

            for record in section_records:
                record["section_idx"] = section_idx
            standardized_excited_records.extend(section_records)

            # account for possibility of multiple excited state sections
            if out.get("excited_states", None) is None:
                out["excited_states"] = [excited_state_data]
            else:
                out["excited_states"].append(excited_state_data)
            continue

        ### CASSCF-like excited state results section ###
        if is_casscf_like_excited_state_header(line):
            section_idx = len(out.get("casscf_energies", []))
            excited_state_data, section_records, i = parse_casscf_excited_state_section(
                lines, start=i, return_standardized_records=True
            )

            for record in section_records:
                record["section_idx"] = section_idx
            standardized_excited_records.extend(section_records)

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
            section_idx = len(out.get("casscf_transition_dipoles", []))
            transition_dipole_data, section_records, i = (
                parse_casscf_transition_dipole_section(
                    lines,
                    start=i,
                    multiplicity=transition_multiplicity,
                    return_standardized_records=True,
                )
            )

            for record in section_records:
                record["section_idx"] = section_idx
            standardized_excited_records.extend(section_records)

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

        ### EOM-CCSD Energies ###
        if EOMCCSD_ENERGIES_HEADER in line:
            section_idx = len(out.get("eomccsd_energies", []))
            eomccsd_energy_data, section_records, i = parse_eomccsd_energies(
                lines, start=i, return_standardized_records=True
            )

            for record in section_records:
                record["section_idx"] = section_idx
            standardized_excited_records.extend(section_records)

            if out.get("eomccsd_energies", None) is None:
                out["eomccsd_energies"] = [eomccsd_energy_data]
            else:
                out["eomccsd_energies"].append(eomccsd_energy_data)
            continue

        ### EOM-CCSD Transition Properties ###
        if EOMCCSD_TRANSITION_PROPERTIES_HEADER in line:
            section_idx = len(out.get("eomccsd_transition", []))
            eomccsd_transition_data, section_records, i = (
                parse_eomccsd_transition_properties(
                    lines, start=i, return_standardized_records=True
                )
            )

            for record in section_records:
                record["section_idx"] = section_idx
            standardized_excited_records.extend(section_records)

            if out.get("eomccsd_transition", None) is None:
                out["eomccsd_transition"] = [eomccsd_transition_data]
            else:
                out["eomccsd_transition"].append(eomccsd_transition_data)
            continue

        ### EOM-CCSD transition dipole matrix elements (<i|mu|j>: x y z) ###
        if is_eomccsd_transition_mu_line(line):
            section_idx = len(out.get("eomccsd_transition_dipoles", []))
            eomccsd_transition_data, section_records, i = (
                parse_eomccsd_transition_mu_elements(
                    lines, start=i, return_standardized_records=True
                )
            )

            for record in section_records:
                record["section_idx"] = section_idx
            standardized_excited_records.extend(section_records)

            if out.get("eomccsd_transition_dipoles", None) is None:
                out["eomccsd_transition_dipoles"] = [eomccsd_transition_data]
            else:
                out["eomccsd_transition_dipoles"].append(eomccsd_transition_data)
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

    out["excited_state_records"] = build_standardized_excited_records(
        standardized_excited_records
    )

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
    roots = sorted(
        {root for section in out["casscf_energies"] for root in section.keys()}
    )
    root_keys = sorted(
        {
            key
            for section in out["casscf_energies"]
            for state_data in section.values()
            for key in state_data.keys()
        }
    )

    o = {}
    for root in roots:
        o[root] = {k: [] for k in root_keys}
        for section in out["casscf_energies"]:
            state_data = section.get(root)
            for k in root_keys:
                o[root][k].append(
                    np.nan if state_data is None else state_data.get(k, np.nan)
                )

    out["casscf_energies"] = o


def parse_eomccsd_energies(
    lines: list, start: int, return_standardized_records: bool = False
) -> dict:
    """Parse EOM-CCSD energy section from terachem output."""
    assert "EOM-CCSD Energies" in lines[start]
    source = inspect.currentframe().f_code.co_name

    def state_line_parser(myline):
        parts = myline.strip().split()
        if len(parts) not in (2, 4):
            raise ValueError(f"Unexpected EOM-CCSD energy line format: {myline}")

        root = int(parts[0])
        total_energy_au = float(parts[1])
        if len(parts) == 4:
            exc_energy_au = float(parts[2])
            exc_energy_ev = float(parts[3])
        elif len(parts) == 2:
            exc_energy_au = float("nan")
            exc_energy_ev = float("nan")
        else:
            raise ValueError(f"Unexpected EOM-CCSD energy line format: {myline}")

        return dict(
            root=root,
            total_energy_au=total_energy_au,
            exc_energy_au=exc_energy_au,
            exc_energy_ev=exc_energy_ev,
        )

    j = start + 1
    out = {}
    standardized_records = []
    while j < len(lines):
        line = lines[j]
        stripped = line.strip()

        if not stripped:
            if out:
                break
            j += 1
            continue

        if (
            stripped.startswith("Root")
            or "EOM-CCSD Energies" in stripped
            or set(stripped) == {"-"}
        ):
            j += 1
            continue

        try:
            state_data = state_line_parser(line)
        except Exception:
            break

        root = state_data.pop("root")
        out[root] = state_data
        standardized_records.append(
            make_standardized_excited_record(
                source=source,
                state_i=0,
                state_j=root,
                total_energy_au=state_data["total_energy_au"],
                exc_energy_au=state_data["exc_energy_au"],
                exc_energy_ev=state_data["exc_energy_ev"],
            )
        )
        j += 1
    if return_standardized_records:
        return out, standardized_records, j
    return out, j


def parse_eomccsd_transition_properties(
    lines: list, start: int, return_standardized_records: bool = False
) -> dict:
    """Parse EOM-CCSD transition properties section from terachem output."""
    assert "EOM-CCSD Transition" in lines[start]
    source = inspect.currentframe().f_code.co_name

    def transition_line_parser(myline):
        transition_match = re.match(r"\s*(\d+)\s*->\s*(\d+)\s+(.*)", myline)
        if transition_match is None:
            raise ValueError(f"Unexpected EOM-CCSD transition line format: {myline}")

        transition_state1 = int(transition_match.group(1))
        transition_state2 = int(transition_match.group(2))
        parts = transition_match.group(3).strip().split()
        transition = f"{transition_state1} -> {transition_state2}"

        if len(parts) == 2:
            return dict(
                transition=transition,
                exc_energy_au=float("nan"),
                exc_energy_ev=float(parts[0]),
                osc_strength=float(parts[1]),
            )

        if len(parts) == 3:
            return dict(
                transition=transition,
                exc_energy_au=float(parts[0]),
                exc_energy_ev=float(parts[1]),
                osc_strength=float(parts[2]),
            )

        if len(parts) == 5:
            raise ValueError(
                "EOM-CCSD transition properties rows with five values are invalid: "
                f"{myline.strip()}"
            )

        raise ValueError(f"Unexpected EOM-CCSD transition line format: {myline}")

    j = start + 1
    out = {}
    standardized_records = []
    while j < len(lines):
        line = lines[j]
        stripped = line.strip()

        if not stripped:
            if out:
                break
            j += 1
            continue

        if (
            stripped.startswith("Transition")
            or "EOM-CCSD Transition" in stripped
            or set(stripped) == {"-"}
        ):
            j += 1
            continue

        if re.match(r"\s*\d+\s*->\s*\d+\s+", line) is None:
            break
        transition_data = transition_line_parser(line)
        transition_match = re.match(r"\s*(\d+)\s*->\s*(\d+)\s+(.*)", line)
        state_i = int(transition_match.group(1))
        state_j = int(transition_match.group(2))

        transition = transition_data.pop("transition")
        out[transition] = transition_data
        standardized_records.append(
            make_standardized_excited_record(
                source=source,
                state_i=state_i,
                state_j=state_j,
                exc_energy_au=transition_data["exc_energy_au"],
                exc_energy_ev=transition_data["exc_energy_ev"],
                osc_strength=transition_data["osc_strength"],
            )
        )
        j += 1
    if return_standardized_records:
        return out, standardized_records, j
    return out, j


def is_eomccsd_transition_mu_line(myline: str) -> bool:
    """Check whether a line contains an EOM transition dipole matrix element."""
    return (
        re.match(r"\s*<\s*\d+\s*\|\s*mu\s*\|\s*\d+\s*>\s*:", myline, re.IGNORECASE)
        is not None
    )


def parse_eomccsd_transition_mu_elements(
    lines: list, start: int, return_standardized_records: bool = False
) -> dict:
    """Parse headerless EOM transition dipoles in '<i|mu|j>: x y z' format."""
    source = inspect.currentframe().f_code.co_name

    def transition_mu_line_parser(myline):
        mu_match = re.match(
            r"\s*<\s*(\d+)\s*\|\s*mu\s*\|\s*(\d+)\s*>\s*:\s*"
            r"([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s+"
            r"([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s+"
            r"([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*$",
            myline,
            re.IGNORECASE,
        )
        if mu_match is None:
            raise ValueError(f"Unexpected EOM transition dipole format: {myline}")

        transition_state1 = int(mu_match.group(1))
        transition_state2 = int(mu_match.group(2))
        Tx = float(mu_match.group(3))
        Ty = float(mu_match.group(4))
        Tz = float(mu_match.group(5))
        T_mag = float(np.sqrt(Tx**2 + Ty**2 + Tz**2))

        return dict(
            transition=f"{transition_state1} -> {transition_state2}",
            Tx=Tx,
            Ty=Ty,
            Tz=Tz,
            T_mag=T_mag,
            osc_strength=float("nan"),
        )

    assert is_eomccsd_transition_mu_line(lines[start])

    j = start
    out = {}
    standardized_records = []
    while j < len(lines):
        line = lines[j]
        stripped = line.strip()

        if not stripped:
            j += 1
            continue

        if not is_eomccsd_transition_mu_line(line):
            if out:
                break
            j += 1
            continue

        transition_data = transition_mu_line_parser(line)
        transition_match = re.match(
            r"\s*<\s*(\d+)\s*\|\s*mu\s*\|\s*(\d+)\s*>\s*:", line
        )
        state_i = int(transition_match.group(1))
        state_j = int(transition_match.group(2))
        transition = transition_data.pop("transition")
        out[transition] = transition_data
        standardized_records.append(
            make_standardized_excited_record(
                source=source,
                state_i=state_i,
                state_j=state_j,
                Tx=transition_data["Tx"],
                Ty=transition_data["Ty"],
                Tz=transition_data["Tz"],
                T_mag=transition_data["T_mag"],
                osc_strength=transition_data["osc_strength"],
            )
        )
        j += 1

    if return_standardized_records:
        return out, standardized_records, j
    return out, j


def parse_casscf_transition_dipole_section(
    lines: list,
    start: int,
    multiplicity: Union[None, str] = None,
    return_standardized_records: bool = False,
) -> dict:
    """Parse a casscf-like transition dipole results section"""
    source = inspect.currentframe().f_code.co_name

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
    standardized_records = []
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

        transition_match = re.match(r"\s*(\d+)\s*->\s*(\d+)\s+(.*)", line)
        state_i = int(transition_match.group(1)) - 1  # is 1-indexed in TC output
        state_j = int(transition_match.group(2)) - 1
        transition = td_data.pop("transition")
        out[transition] = td_data
        standardized_records.append(
            make_standardized_excited_record(
                source=source,
                state_i=state_i,
                state_j=state_j,
                multiplicity=multiplicity,
                Tx=td_data["Tx"],
                Ty=td_data["Ty"],
                Tz=td_data["Tz"],
                T_mag=td_data["T_mag"],
                osc_strength=td_data["osc_strength"],
            )
        )
        j += 1

    if multiplicity is None:
        section = out
    else:
        section = {"multiplicity": multiplicity, "transitions": out}

    if return_standardized_records:
        return section, standardized_records, j
    return section, j


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


def parse_casscf_excited_state_section(
    lines: list, start: int, return_standardized_records: bool = False
) -> dict:
    """Parse a casscf-like excited state results section"""
    source = inspect.currentframe().f_code.co_name

    def check_end(myline):
        return not myline.strip()

    def excited_state_line_parser(myline):
        parts = myline.strip().split()
        if len(parts) < 3:
            raise ValueError(f"Unexpected CASSCF-like energy line format: {myline}")

        root = int(parts[0])
        mult_string = parts[1]  # 'singlet', 'triplet', etc
        float_values = [float(x) for x in parts[2:]]
        total_energy_au = float_values[0]
        extras = float_values[1:]

        if len(extras) == 0:
            # Root 1 is ground state--no excitation information
            exc_energy_au = float("nan")
            exc_energy_ev = float("nan")
            exc_energy_nm = float("nan")
            osc_strength = float("nan")
        elif len(extras) == 2:
            # Roots > 1 have (au, eV)
            exc_energy_au = extras[0]
            exc_energy_ev = extras[1]
            exc_energy_nm = float("nan")
            osc_strength = float("nan")
        elif len(extras) == 4:
            # HH-TDA-like formatting adds (nm, osc)
            exc_energy_au = extras[0]
            exc_energy_ev = extras[1]
            exc_energy_nm = extras[2]
            osc_strength = extras[3]
        else:
            raise ValueError(
                "Unexpected CASSCF-like energy line format "
                f"(got {len(parts)} columns): {myline}"
            )

        out = dict(
            root=root,
            multiplicity=mult_string,
            total_energy_au=total_energy_au,
            exc_energy_au=exc_energy_au,
            exc_energy_ev=exc_energy_ev,
            exc_energy_nm=exc_energy_nm,
            osc_strength=osc_strength,
        )

        return out

    j = start + 1
    out = {}
    standardized_records = []
    while j < len(lines):
        line = lines[j]
        stripped = line.strip()

        if check_end(line):
            if out:
                break
            j += 1
            continue

        if stripped.lower().startswith("root") or set(stripped) == {"-"}:
            j += 1
            continue

        try:
            state_data = excited_state_line_parser(line)
        except ValueError as e:
            print(
                f"WARNING: Breaking parse of section due to error: {e} (line {j}: {lines[j]})"
            )
            break

        root = state_data.pop("root")
        out[root] = state_data
        standardized_records.append(
            make_standardized_excited_record(
                source=source,
                state_i=0,
                state_j=root - 1,  # roots here are 1-indexed in TC output
                multiplicity=state_data["multiplicity"],
                total_energy_au=state_data["total_energy_au"],
                exc_energy_au=state_data["exc_energy_au"],
                exc_energy_ev=state_data["exc_energy_ev"],
                exc_energy_nm=state_data["exc_energy_nm"],
                osc_strength=state_data["osc_strength"],
            )
        )
        j += 1

    if return_standardized_records:
        return out, standardized_records, j
    return out, j


def parse_excited_state_section(
    lines: list, start: int, return_standardized_records: bool = False
) -> dict:
    """Parse an excited state results section from terachem output.

    Args:
        lines (list): List of lines from the terachem output file.
        start (int): Index of the line where the excited state section starts.
    """
    assert "Final Excited State Results" in lines[start]
    source = inspect.currentframe().f_code.co_name

    def check_end(myline):
        return myline.isspace()

    def excited_state_line_parser(myline):
        parts = myline.strip().split()
        out = {
            "root": int(parts[0]),
            "total_energy_au": float(parts[1]),
            "exc_energy_ev": float(parts[2]),
            "osc_strength": float(parts[3]),
            "s_squared": float(parts[4]),
            "max_ci_coeff": float(parts[5]),
        }
        return out

    j = start + 4

    out = {}
    standardized_records = []
    while not check_end(lines[j]):
        state_data = excited_state_line_parser(lines[j])
        root = state_data.pop("root")
        out[root] = state_data

        standardized_records.append(
            make_standardized_excited_record(
                source=source,
                state_i=0,
                state_j=root,
                total_energy_au=state_data["total_energy_au"],
                # "Final Excited State Results" reports excitation energies in eV.
                exc_energy_ev=state_data["exc_energy_ev"],
                osc_strength=state_data["osc_strength"],
                s_squared=state_data["s_squared"],
                max_ci_coeff=state_data["max_ci_coeff"],
            )
        )
        j += 1

    if return_standardized_records:
        return out, standardized_records, j
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
