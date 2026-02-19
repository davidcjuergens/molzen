"""IO utils for MO data"""

import numpy as np
import subprocess
import glob
import os


def get_occupied_ntos(molden_path, occ_thresh=0.1):
    """Extract indices of occupied from a .molden file.

    Args:
        molden_path (str): Path to .molden file
        occ_thresh (float): Occupation threshold for determining occupied orbitals
    """
    occs = []
    with open(molden_path, "r") as f:
        for l in f:
            if "Occup=" in l:
                occ = float(l.split()[1])
                occs.append(occ)
    occs = np.array(occs)
    occupied_indices = np.where(np.abs(occs) > occ_thresh)[0]
    return occupied_indices + 1, occs[occupied_indices]


def generate_nto_cubes(
    molden_path,
    occ_thresh=0.1,
    multiwfn_path="Multiwfn",
    grid_quality="2",
    output_mode="1",
):
    """
    Automates Multiwfn to generate .cub files.

    Args:
        molden_path (str): Path to .molden file
        occ_thresh (float): Occupation threshold in NTO for creating .cub files
        multiwfn_path (str): Path to Multiwfn executable
        grid_quality (str): Grid quality for cube generation: 1= Low, 2 = Medium, 3 = High
        output_mode (str): Output mode for cube files: '1' for separate files, '2' for single file with all orbitals
    """

    occ_indices_1idx, occs = get_occupied_ntos(molden_path, occ_thresh=occ_thresh)
    assert np.sum(occs > 0) > 1 and np.sum(occs < 0) > 1, (
        "Need at least 2 occupied orbitals, one with positive occupation and one with negative occupation."
    )

    # Construct the input string with explicit newlines.
    # We use a raw string or explicit \n to ensure clean parsing.
    # Note: 'output_mode' should be the INTEGER corresponding to the menu option.
    input_commands = [
        "200",  # Other functions (Part 2)
        "3",  # Function: Generate cube file for orbitals
        ",".join(map(str, occ_indices_1idx)),  # Orbital selection
        str(grid_quality),  # Grid quality (e.g., '2')
        str(output_mode),  # Output mode (Must be integer! e.g., '2' for separate)
        "0",  # Return to main menu (good practice to close gracefully)
        "q",  # Exit program
    ]

    # Join with newlines
    multiwfn_input = "\n".join(input_commands) + "\n"
    workdir = os.path.dirname(molden_path)

    result = subprocess.run(
        [multiwfn_path, molden_path],
        input=multiwfn_input,
        text=True,  # Handles string encoding automatically
        capture_output=True,  # Captures stdout/stderr for debugging
        check=True,  # Raises error if exit code != 0
        cwd=workdir,  # Run in the directory of the molden file
    )

    # Now rename the geneated files with useful information.
    prefix = os.path.basename(molden_path).replace(".molden", "")
    if prefix.startswith("orb"):
        print(
            """Warning: Molden file name starts with 'orb' and this function looks for files that match 'orb*.cub' in the renaming step. 
            Consider renaming the molden file to avoid this.
            """
        )

    cubfiles = list(sorted(glob.glob(os.path.join(workdir, "orb*.cub"))))

    # Dont clobber other existing .cub files in the directory..
    assert len(cubfiles) == len(occ_indices_1idx), (
        f"Number of orb*.cub files in directory ({len(cubfiles)}) does not match number of occupied orbitals ({len(occ_indices_1idx)})."
    )

    def occ2tag(occ):
        return "hole" if occ < 0 else "part"

    for i, cubfile in enumerate(cubfiles):
        occ = occs[i]
        absocc = abs(occ)
        absolute_idx = occ_indices_1idx[i]
        nto_idx = (absolute_idx + 1) // 2
        hptag = occ2tag(occ)

        new_name = f"{prefix}.nto{nto_idx}_{hptag}_occ_{absocc:.3f}.cub"
        new_path = os.path.join(workdir, new_name)
        os.rename(cubfile, new_path)
