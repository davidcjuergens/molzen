"""Parsing terachem outputs"""

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
        if "Processed Input file:" in line:
            assert not out.get("input_args"), "Input args already parsed!"

            tc_kwargs, i = parse_tc_input_flags(lines, start=i)

            out["input_args"] = tc_kwargs
            continue

        ### Ground State Results ###
        if "FINAL ENERGY:" in line:
            energy_line = line.strip().split()
            out["ground_state_energy"] = float(energy_line[2])
            i += 1
            continue

        #### Excited State Results ###
        if "Final Excited State Results" in line:
            excited_state_data, i = parse_excited_state_section(lines, start=i)

            # account for possibility of multiple excited state sections
            if out.get("excited_states", None) is None:
                out["excited_states"] = {1: excited_state_data}
            else:
                cur_section = len(out["excited_states"]) + 1
                out["excited_states"][cur_section] = excited_state_data
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

    return out


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
