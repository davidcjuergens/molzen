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
    "Dx": np.nan,
    "Dy": np.nan,
    "Dz": np.nan,
    "D_mag": np.nan,
    "Px": np.nan,
    "Py": np.nan,
    "Pz": np.nan,
    "P_mag": np.nan,
    "velocity_osc_strength": np.nan,
    "Lx": np.nan,
    "Ly": np.nan,
    "Lz": np.nan,
    "L_mag": np.nan,
    "rot_strength": np.nan,
    "Qxy": np.nan,
    "Qxz": np.nan,
    "Qyz": np.nan,
    "Qxx": np.nan,
    "Qyy": np.nan,
    "Qzz": np.nan,
    "R2": np.nan,
    "Q_norm": np.nan,
    "origin_x": np.nan,
    "origin_y": np.nan,
    "origin_z": np.nan,
    "s_squared": np.nan,
    "max_ci_coeff": np.nan,
    "energy_gradient": None,
    "nonadiabatic_coupling": None,
    "multiplicity_i": None,
    "multiplicity_j": None,
    "row_S": None,
    "row_Sz": None,
    "col_S": None,
    "col_Sz": None,
    "soc": None,
    "soc_units": None,
}

_MULTIPLICITY_NAMES = {
    1: "singlet",
    2: "doublet",
    3: "triplet",
    4: "quartet",
    5: "quintet",
    6: "sextet",
    7: "septet",
    8: "octet",
    9: "nonet",
    10: "decet",
}


def infer_multiplicity_from_s_squared(
    s_squared: float | int | str | None, *, tolerance: float = 0.25
) -> str | None:
    """Infer a spin multiplicity label from <S^2>, if it is close to valid spin."""
    try:
        value = float(s_squared)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value) or value < 0:
        return None

    multiplicity = int(round(np.sqrt(1.0 + 4.0 * value)))
    if multiplicity < 1:
        return None

    spin = (multiplicity - 1) / 2.0
    expected_s_squared = spin * (spin + 1.0)
    if abs(value - expected_s_squared) > tolerance:
        return None
    return _MULTIPLICITY_NAMES.get(multiplicity, f"{multiplicity}-plet")


def make_standardized_excited_record(**kwargs) -> dict:
    """Create one standardized excited-state record with defaults for missing fields."""
    unknown_keys = sorted(
        set(kwargs.keys()) - set(STANDARDIZED_EXCITED_RECORD_DEFAULTS)
    )
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


#### Some constants ####
INPUT_ARGS_HEADER = "Processed Input file:"
SCF_ENERGY_HEADER = "FINAL ENERGY:"
EXCITED_STATES_RESULTS_HEADER = "Final Excited State Results"
CASSCF_SINGLET_TRANSITION_DIPOLE_HEADER = "Singlet state electronic transitions:"
CASSCF_TRIPLET_TRANSITION_DIPOLE_HEADER = "Triplet state electronic transitions:"
_STATE_DIPOLE_HEADER_RE = re.compile(r"^\s*([A-Za-z]+)\s+state dipole moments:\s*$")
_CASSCF_PROPERTY_HEADERS = (
    (
        "casscf_velocity_transition_dipoles",
        re.compile(r"^\s*([A-Za-z]+)\s+state velocity transition dipole moments:\s*$"),
    ),
    (
        "casscf_magnetic_transition_dipoles",
        re.compile(
            r"^\s*([A-Za-z]+)\s+state magnetic transition dipole moments"
            r" and rotational strengths:\s*$"
        ),
    ),
    (
        "casscf_transition_quadrupoles",
        re.compile(
            r"^\s*([A-Za-z]+)\s+state electric transition quadrupole"
            r" moments \(Cartesian\):\s*$"
        ),
    ),
    (
        "casscf_state_quadrupoles",
        re.compile(
            r"^\s*([A-Za-z]+)\s+state electric quadrupole moments \(Cartesian\):\s*$"
        ),
    ),
)
_TRANSITION_PROPERTY_COLUMNS = {
    "casscf_velocity_transition_dipoles": (
        "Px",
        "Py",
        "Pz",
        "P_mag",
        "velocity_osc_strength",
    ),
    "casscf_magnetic_transition_dipoles": (
        "Lx",
        "Ly",
        "Lz",
        "L_mag",
        "rot_strength",
    ),
    "casscf_transition_quadrupoles": (
        "Qxy",
        "Qxz",
        "Qyz",
        "Qxx",
        "Qyy",
        "Qzz",
        "Q_norm",
    ),
}
_ROOT_PROPERTY_COLUMNS = {
    "casscf_state_quadrupoles": (
        "Qxy",
        "Qxz",
        "Qyz",
        "Qxx",
        "Qyy",
        "Qzz",
        "R2",
    ),
}
_PROPERTY_RECORD_SOURCES = {
    "casscf_velocity_transition_dipoles": (
        "parse_casscf_velocity_transition_dipole_section"
    ),
    "casscf_magnetic_transition_dipoles": (
        "parse_casscf_magnetic_transition_dipole_section"
    ),
    "casscf_state_quadrupoles": "parse_casscf_state_quadrupole_section",
    "casscf_transition_quadrupoles": "parse_casscf_transition_quadrupole_section",
}
_TRANSITION_ROW_RE = re.compile(r"^\s*(\d+)\s*->\s*(\d+)\s+(.*)$")
CASSCF_ORB_ENERGIES_HEADER = "Orbital      Energy"
EOMCCSD_ENERGIES_HEADER = "====> EOM-CCSD Energies <===="
EOMCCSD_TRANSITION_PROPERTIES_HEADER = "====> EOM-CCSD Transition Properties <===="
GRADIENT_HEADER = "Gradient units are Hartree/Bohr"
FMS_COORDINATES_HEADER = "Receiving coordinates from FMS"
_SOC_HEADER_RE = re.compile(
    r"^\s*([A-Za-z]+)\[(\d+)\]\s*-\s*([A-Za-z]+)\[(\d+)\]\s+"
    r"Spin-orbit coupling\s*\(([^)]+)\):\s*$"
)
_SPIN_PAIR_RE = re.compile(
    r"\(\s*([+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)\s*,\s*"
    r"([+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)\s*\)"
)
_COMPLEX_RE = re.compile(
    r"([+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)\s*([+-])\s*"
    r"([+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)i"
)

MINIMIZE_CONVERGED_HEADER = "Converged!"
#### End constants ####


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

    if not raw_str_in:
        with open(file_path, "r") as tc_file:
            lines = tc_file.readlines()
    else:
        lines = file_path.splitlines()

    running_minimization = False
    out = {"minimization_converged": False}

    standardized_excited_records = []
    i = 0
    while i < len(lines):
        line = lines[i]

        ### Input Arguments ###
        if INPUT_ARGS_HEADER in line:
            assert not out.get("input_args"), "Input args already parsed!"

            tc_kwargs, i = parse_tc_input_flags(lines, start=i)

            out["input_args"] = tc_kwargs

            # check run type
            if tc_kwargs["run"] in ("minimize"):
                running_minimization = True

            continue

        ### Coordinates sent by the FMS interface ###
        if FMS_COORDINATES_HEADER in line:
            frame, i = parse_fms_coordinate_section(lines, start=i)
            frames = out.setdefault("fms_coordinates", [])
            if frames and frame["elements"] != frames[0]["elements"]:
                raise ValueError("Element order changed between FMS coordinate frames.")
            frames.append(frame)
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

        ### Minimization convergence status (not guaranteed to be present in all outputs) ###
        if MINIMIZE_CONVERGED_HEADER in line:
            if running_minimization:
                out["minimization_converged"] = True
            else:
                # possible this occurs if we are missing kwarg aliases for "minimize", e.g., "optimize"
                # if so, update
                print(
                    f"WARNING: Found '{MINIMIZE_CONVERGED_HEADER}' in output but did not detect minimization in job."
                )
                print("Setting minimization_converged to True anyways.")
                out["minimization_converged"] = True

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

        ### CASSCF-like state dipole moments (permanent dipoles, Debye) ###
        state_dipole_spin = multiplicity_from_state_dipole_header(line)
        if state_dipole_spin is not None:
            parsed_state_dipoles = out.get("casscf_state_dipoles", [])
            section_idx = sum(
                section.get("multiplicity") == state_dipole_spin
                for section in parsed_state_dipoles
            )
            state_dipole_data, section_records, i = parse_casscf_state_dipole_section(
                lines,
                start=i,
                multiplicity=state_dipole_spin,
                return_standardized_records=True,
            )

            for record in section_records:
                record["section_idx"] = section_idx
            standardized_excited_records.extend(section_records)

            if out.get("casscf_state_dipoles", None) is None:
                out["casscf_state_dipoles"] = [state_dipole_data]
            else:
                out["casscf_state_dipoles"].append(state_dipole_data)
            continue

        ### Velocity dipoles, magnetic dipoles, and Cartesian quadrupoles ###
        property_header = casscf_property_section_header(line)
        if property_header is not None:
            property_key, property_spin = property_header
            property_data, section_records, i = parse_casscf_property_section(
                lines,
                start=i,
                kind=property_key,
                multiplicity=property_spin,
                return_standardized_records=True,
            )
            _append_multiplicity_section(
                out,
                property_key,
                property_data,
                section_records,
                standardized_excited_records,
            )
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

        ### Gradients
        if GRADIENT_HEADER in line:
            gradient_data, i = parse_gradient_section(lines, start=i)
            input_args = out.get("input_args", {})
            run_type = str(input_args.get("run", "")).strip().lower()
            section_idx = sum(
                record["source"] == "parse_gradient_section"
                for record in standardized_excited_records
            )

            if run_type == "gradient":
                standardized_excited_records.append(
                    make_standardized_excited_record(
                        source="parse_gradient_section",
                        section_idx=section_idx,
                        state_i=0,
                        state_j=int(input_args["cistarget"]),
                        energy_gradient=gradient_data["gradient"],
                    )
                )
            elif run_type == "coupling":
                standardized_excited_records.append(
                    make_standardized_excited_record(
                        source="parse_gradient_section",
                        section_idx=section_idx,
                        state_i=int(input_args["nacstate1"]),
                        state_j=int(input_args["nacstate2"]),
                        nonadiabatic_coupling=gradient_data["gradient"],
                    )
                )
            continue

        ### Spin-orbit coupling matrices between indexed states ###
        if soc_block_header(line) is not None:
            soc_block, section_records, i = parse_soc_section(
                lines, start=i, return_standardized_records=True
            )
            parsed_soc = out.get("spin_orbit_couplings", [])
            section_idx = sum(
                item["multiplicity_i"] == soc_block["multiplicity_i"]
                and item["state_i"] == soc_block["state_i"]
                and item["multiplicity_j"] == soc_block["multiplicity_j"]
                and item["state_j"] == soc_block["state_j"]
                for item in parsed_soc
            )
            for record in section_records:
                record["section_idx"] = section_idx
            standardized_excited_records.extend(section_records)
            out.setdefault("spin_orbit_couplings", []).append(soc_block)
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
    if out.get("casscf_state_dipoles") is not None:
        postprocess_casscf_state_dipoles(out)
    for property_key, sort_labels in (
        ("casscf_velocity_transition_dipoles", False),
        ("casscf_magnetic_transition_dipoles", False),
        ("casscf_transition_quadrupoles", False),
        ("casscf_state_quadrupoles", True),
    ):
        if out.get(property_key) is not None:
            postprocess_casscf_property_sections(
                out, property_key, sort_labels=sort_labels
            )
    if out.get("casscf_orbitals") is not None:
        postprocess_casscf_orbitals(out)
    if out.get("fms_coordinates") is not None:
        postprocess_fms_coordinates(out)
    if out.get("spin_orbit_couplings") is not None:
        postprocess_spin_orbit_couplings(out)
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


def multiplicity_from_state_dipole_header(myline: str) -> str | None:
    """Return the spin label from a 'Singlet state dipole moments:' style header."""
    match = _STATE_DIPOLE_HEADER_RE.match(myline)
    if match is None:
        return None
    return match.group(1).lower()


def casscf_property_section_header(myline: str) -> tuple[str, str] | None:
    """Return (output key, multiplicity) for a spin-labeled property header."""
    for key, pattern in _CASSCF_PROPERTY_HEADERS:
        match = pattern.match(myline)
        if match is not None:
            return key, match.group(1).lower()
    return None


def _append_multiplicity_section(
    out: dict,
    key: str,
    section: dict,
    section_records: list,
    standardized_records: list,
) -> None:
    """Store one spin section and align its records with earlier sections of that spin."""
    section_idx = sum(
        item.get("multiplicity") == section["multiplicity"] for item in out.get(key, [])
    )
    for record in section_records:
        record["section_idx"] = section_idx
    standardized_records.extend(section_records)
    out.setdefault(key, []).append(section)


def _merge_labeled_maps(grouped: dict, sort_labels: bool = False) -> dict:
    """Stack repeated {label: {field: value}} maps into {label: {field: [values]}}."""
    merged = {}
    for group, sections in grouped.items():
        labels = []
        field_names = []
        for section in sections:
            for label, values in section.items():
                if label not in labels:
                    labels.append(label)
                for field in values:
                    if field not in field_names:
                        field_names.append(field)
        if sort_labels:
            labels = sorted(labels)
            field_names = sorted(field_names)

        merged[group] = {}
        for label in labels:
            merged[group][label] = {field: [] for field in field_names}
            for section in sections:
                values = section.get(label)
                for field in field_names:
                    merged[group][label][field].append(
                        np.nan if values is None else values.get(field, np.nan)
                    )
    return merged


def postprocess_casscf_property_sections(out, key: str, sort_labels: bool = False):
    """Group a spin-labeled property by multiplicity, then by root or transition."""
    grouped = {}
    for section in out[key]:
        grouped.setdefault(section["multiplicity"], []).append(section["rows"])
    out[key] = _merge_labeled_maps(grouped, sort_labels=sort_labels)


def postprocess_casscf_state_dipoles(out):
    """Organize state dipole sections by multiplicity and root.

    Each printed root keeps the 1-based index from the output. Repeated sections
    (for example along an optimization) are stacked into lists, with missing
    roots filled by NaN.
    """
    grouped = {}
    for section in out["casscf_state_dipoles"]:
        grouped.setdefault(section["multiplicity"], []).append(section["dipoles"])
    out["casscf_state_dipoles"] = _merge_labeled_maps(grouped, sort_labels=True)


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


def parse_fms_coordinate_section(lines: list, start: int) -> tuple[dict, int]:
    """Parse one 'Receiving coordinates from FMS' geometry.

    Atom lines are element symbol plus x, y, z. The printed coordinates are
    Angstrom. Parsing stops at a blank line or the next non-coordinate line.
    """
    assert FMS_COORDINATES_HEADER in lines[start]

    elements = []
    coords = []
    j = start + 1
    while j < len(lines):
        stripped = lines[j].strip()
        if not stripped:
            if coords:
                break
            j += 1
            continue

        parts = stripped.split()
        if len(parts) != 4 or not parts[0][:1].isalpha():
            if coords:
                break
            raise ValueError(f"Unexpected FMS coordinate line: {lines[j]}")

        try:
            xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
        except ValueError:
            if coords:
                break
            raise ValueError(f"Unexpected FMS coordinate line: {lines[j]}") from None

        elements.append(parts[0])
        coords.append(xyz)
        j += 1

    if not coords:
        raise ValueError("FMS coordinate section contained no atoms.")

    return {"elements": elements, "xyz": np.asarray(coords, dtype=float)}, j


def postprocess_fms_coordinates(out):
    """Stack FMS geometries into one element list and an (n_frames, n_atoms, 3) array."""
    frames = out["fms_coordinates"]
    out["fms_coordinates"] = {
        "elements": frames[0]["elements"],
        "xyz": np.stack([frame["xyz"] for frame in frames], axis=0),
        "units": "angstrom",
    }


def parse_gradient_section(lines: list, start: int) -> tuple[dict, int]:
    """Parse a gradient section from terachem output.
    Note: This could be a state energy gradient, or it could be a NAC vector..
    """
    assert GRADIENT_HEADER in lines[start]
    assert "------" in lines[start + 1]
    assert "dE/dX" in lines[start + 2]

    x, y, z = [], [], []
    j = start + 3

    def check_end(myline):
        return "------" in myline  # the end of the grad section seems to have ---

    while not check_end(lines[j]):
        parts = lines[j].strip().split()
        x.append(float(parts[0]))
        y.append(float(parts[1]))
        z.append(float(parts[2]))
        j += 1

    grad = np.array([x, y, z]).T
    return {"gradient": grad}, j


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


def parse_casscf_state_dipole_section(
    lines: list,
    start: int,
    multiplicity: str,
    return_standardized_records: bool = False,
) -> dict:
    """Parse a state-resolved permanent dipole section.

    Expected layout (values are Debye, roots are 1-indexed as printed):

        Singlet state dipole moments:

          Root       Dx         Dy         Dz         |D|  (Debye)
        ------------------------------------------------------------
             1    -1.7867    -0.0581     0.0072     1.7877
    """
    source = inspect.currentframe().f_code.co_name
    assert multiplicity_from_state_dipole_header(lines[start]) == multiplicity

    def state_dipole_line_parser(myline):
        parts = myline.strip().split()
        if len(parts) != 5:
            raise ValueError(f"Unexpected state dipole line format: {myline}")

        return dict(
            root=int(parts[0]),
            Dx=float(parts[1]),
            Dy=float(parts[2]),
            Dz=float(parts[3]),
            D_mag=float(parts[4]),
        )

    def is_dipole_header_line(myline):
        stripped = myline.strip().lower()
        return stripped.startswith("root") or set(stripped) == {"-"}

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

        if is_dipole_header_line(line):
            j += 1
            continue

        try:
            dipole_data = state_dipole_line_parser(line)
        except (TypeError, ValueError):
            break

        root = dipole_data.pop("root")
        out[root] = dipole_data
        standardized_records.append(
            make_standardized_excited_record(
                source=source,
                state_i=root - 1,  # roots are 1-indexed in TC output
                state_j=root - 1,
                multiplicity=multiplicity,
                Dx=dipole_data["Dx"],
                Dy=dipole_data["Dy"],
                Dz=dipole_data["Dz"],
                D_mag=dipole_data["D_mag"],
            )
        )
        j += 1

    section = {"multiplicity": multiplicity, "dipoles": out}
    if return_standardized_records:
        return section, standardized_records, j
    return section, j


def _parse_operator_origin(myline: str) -> tuple[float, float, float]:
    """Parse 'Operator origin (a.u.): x, y, z'."""
    try:
        _, values = myline.split(":", 1)
        parts = [part.strip() for part in values.split(",")]
        if len(parts) != 3:
            raise ValueError
        return tuple(float(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"Unexpected operator origin format: {myline}") from exc


def _parse_component_row(myline: str, columns: tuple[str, ...], indexed_by: str):
    """Parse one root or transition row into components named by columns."""
    if indexed_by == "transition":
        match = _TRANSITION_ROW_RE.match(myline)
        if match is None:
            raise ValueError(f"Unexpected transition row format: {myline}")
        state_i = int(match.group(1))
        state_j = int(match.group(2))
        label = f"{state_i} -> {state_j}"
        parts = match.group(3).split()
    elif indexed_by == "root":
        parts = myline.strip().split()
        if not parts:
            raise ValueError(f"Unexpected root row format: {myline}")
        state_i = state_j = int(parts[0])
        label = state_i
        parts = parts[1:]
    else:
        raise ValueError(f"Unknown property index '{indexed_by}'")

    if len(parts) != len(columns):
        raise ValueError(f"Unexpected property row format: {myline}")

    values = {name: float(part) for name, part in zip(columns, parts)}
    return label, state_i, state_j, values


def parse_casscf_property_section(
    lines: list,
    start: int,
    kind: str,
    multiplicity: str,
    return_standardized_records: bool = False,
):
    """Parse one spin-labeled velocity, magnetic, or quadrupole section.

    Transition tables are keyed by the printed 'i -> j' label. State quadrupole
    tables are keyed by the printed 1-based root. Component values are kept in the
    units printed by TeraChem. State quadrupoles also store the operator origin.
    """
    header = casscf_property_section_header(lines[start])
    assert header == (kind, multiplicity)
    source = _PROPERTY_RECORD_SOURCES[kind]
    if kind in _TRANSITION_PROPERTY_COLUMNS:
        columns = _TRANSITION_PROPERTY_COLUMNS[kind]
        indexed_by = "transition"
    else:
        columns = _ROOT_PROPERTY_COLUMNS[kind]
        indexed_by = "root"

    j = start + 1
    rows = {}
    origin = None
    standardized_records = []
    while j < len(lines):
        line = lines[j]
        stripped = line.strip()

        if not stripped:
            if rows:
                break
            j += 1
            continue

        lowered = stripped.lower()
        if (
            lowered.startswith("root")
            or lowered.startswith("transition")
            or set(stripped) == {"-"}
        ):
            j += 1
            continue

        if lowered.startswith("operator origin"):
            origin = _parse_operator_origin(line)
            j += 1
            continue

        try:
            label, state_i, state_j, values = _parse_component_row(
                line, columns, indexed_by
            )
        except (TypeError, ValueError):
            break

        rows[label] = values
        standardized_records.append(
            make_standardized_excited_record(
                source=source,
                state_i=state_i - 1,  # roots and transitions are 1-indexed in TC output
                state_j=state_j - 1,
                multiplicity=multiplicity,
                **values,
            )
        )
        j += 1

    if origin is not None:
        for values in rows.values():
            values["origin_x"], values["origin_y"], values["origin_z"] = origin
        for record in standardized_records:
            record["origin_x"], record["origin_y"], record["origin_z"] = origin

    section = {"multiplicity": multiplicity, "rows": rows}
    if return_standardized_records:
        return section, standardized_records, j
    return section, j


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
        return not myline.strip()

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
                multiplicity=infer_multiplicity_from_s_squared(state_data["s_squared"]),
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


def soc_block_header(myline: str) -> dict | None:
    """Return the two states named by a spin-orbit coupling header, if present."""
    match = _SOC_HEADER_RE.match(myline)
    if match is None:
        return None
    return {
        "multiplicity_i": match.group(1).lower(),
        "state_i": int(match.group(2)),
        "multiplicity_j": match.group(3).lower(),
        "state_j": int(match.group(4)),
        "units": match.group(5).strip(),
    }


def _complex_matrix_element(real_text: str, operator: str, imag_text: str) -> complex:
    """Combine a printed 'a ± bi' element, including forms like '+  -0.0000i'."""
    real = float(real_text)
    imag = float(imag_text)
    if imag_text[0] not in "+-" and operator == "-":
        imag = -imag
    return complex(real, imag)


def parse_soc_section(
    lines: list,
    start: int,
    return_standardized_records: bool = False,
):
    """Parse one spin-orbit coupling matrix.

    The header indices are already 0-based, so Triplet[1] is state_j=1. Rows are
    the (S, Sz) components of the first state and columns are those of the second.
    Values are complex and are kept in the printed unit, usually cm-1. Triplet
    blocks are 3x3 and singlet-triplet blocks are 1x3.
    """
    header = soc_block_header(lines[start])
    assert header is not None
    source = inspect.currentframe().f_code.co_name

    j = start + 1
    col_s = None
    col_sz = None
    row_s = []
    row_sz = []
    matrix_rows = []
    while j < len(lines):
        line = lines[j]
        stripped = line.strip()
        if not stripped:
            if matrix_rows:
                break
            j += 1
            continue
        if set(stripped) == {"-"}:
            j += 1
            continue
        if stripped.startswith("(S"):
            pairs = _SPIN_PAIR_RE.findall(stripped)
            if not pairs:
                raise ValueError(f"Spin-orbit column header has no (S, Sz): {line}")
            col_s = np.asarray([float(spin) for spin, _ in pairs], dtype=float)
            col_sz = np.asarray([float(sz) for _, sz in pairs], dtype=float)
            j += 1
            continue

        pairs = _SPIN_PAIR_RE.findall(stripped)
        components = _COMPLEX_RE.findall(stripped)
        if len(pairs) != 1 or not components:
            if matrix_rows:
                break
            raise ValueError(f"Unexpected spin-orbit row: {line}")
        if col_s is None or len(components) != len(col_s):
            raise ValueError(f"Spin-orbit row does not match the column header: {line}")

        row_s.append(float(pairs[0][0]))
        row_sz.append(float(pairs[0][1]))
        matrix_rows.append(
            [
                _complex_matrix_element(real, operator, imag)
                for real, operator, imag in components
            ]
        )
        j += 1

    if not matrix_rows or col_s is None or col_sz is None:
        raise ValueError(f"Spin-orbit section contained no matrix: {lines[start]}")

    row_s = np.asarray(row_s, dtype=float)
    row_sz = np.asarray(row_sz, dtype=float)
    soc = np.asarray(matrix_rows, dtype=complex)
    block = {
        "multiplicity_i": header["multiplicity_i"],
        "state_i": header["state_i"],
        "multiplicity_j": header["multiplicity_j"],
        "state_j": header["state_j"],
        "units": header["units"],
        "row_S": row_s,
        "row_Sz": row_sz,
        "col_S": col_s,
        "col_Sz": col_sz,
        "soc": soc,
    }
    standardized_records = [
        make_standardized_excited_record(
            source=source,
            state_i=header["state_i"],
            state_j=header["state_j"],
            multiplicity_i=header["multiplicity_i"],
            multiplicity_j=header["multiplicity_j"],
            row_S=row_s,
            row_Sz=row_sz,
            col_S=col_s,
            col_Sz=col_sz,
            soc=soc,
            soc_units=header["units"],
        )
    ]
    if return_standardized_records:
        return block, standardized_records, j
    return block, j


def postprocess_spin_orbit_couplings(out):
    """Group SOC blocks by the two states and stack repeated geometries."""
    blocks = out["spin_orbit_couplings"]
    units = blocks[0]["units"]
    grouped = {}
    for block in blocks:
        if block["units"] != units:
            raise ValueError(
                f"Inconsistent spin-orbit units: {block['units']!r} and {units!r}."
            )
        pair_group = f"{block['multiplicity_i']} -> {block['multiplicity_j']}"
        pair = f"{block['state_i']} -> {block['state_j']}"
        grouped.setdefault(pair_group, {}).setdefault(pair, []).append(block)

    couplings = {}
    for pair_group, pairs in grouped.items():
        couplings[pair_group] = {}
        for pair, repeats in pairs.items():
            first = repeats[0]
            for block in repeats[1:]:
                for key in ("row_S", "row_Sz", "col_S", "col_Sz"):
                    if not np.allclose(block[key], first[key]):
                        raise ValueError(
                            "Spin-orbit (S, Sz) labels changed for "
                            f"{pair_group} {pair}."
                        )
            couplings[pair_group][pair] = {
                "state_i": first["state_i"],
                "state_j": first["state_j"],
                "multiplicity_i": first["multiplicity_i"],
                "multiplicity_j": first["multiplicity_j"],
                "row_S": first["row_S"],
                "row_Sz": first["row_Sz"],
                "col_S": first["col_S"],
                "col_Sz": first["col_Sz"],
                "soc": np.stack([block["soc"] for block in repeats], axis=0),
            }

    out["spin_orbit_couplings"] = {"units": units, "couplings": couplings}


def soc_parser(lines: list, start: int, return_standardized_records: bool = False):
    """Parse one modern spin-orbit coupling block.

    Replaces the retired scalar parser that read 'SOC between' / 'SOC Constant'.
    """
    return parse_soc_section(
        lines, start, return_standardized_records=return_standardized_records
    )
