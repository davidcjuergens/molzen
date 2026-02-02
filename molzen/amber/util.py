import os
import subprocess
import tempfile
import numpy as np
import pytraj as pt


def make_spherical_water_droplet(
    prmtop,
    rst7,
    n_closest,
    solute_sel,
    parmout_path=None,
    cpptraj_path="cpptraj",
):
    """
    Use cpptraj to build a spherical droplet by keeping the closest n_closest solvent
    molecules around the solute selection in a single-frame rst7.

    Args:
        prmtop (str): Path to the topology file.
        rst7 (str): Path to the single-frame restart file.
        n_closest (int): Number of closest solvent molecules to keep.
        solute_sel (str): cpptraj selection for solute.
        parmout_path (str, optional): If provided, write a prmtop for the droplet.
        cpptraj_path (str, optional): Path to cpptraj executable.
    Returns:
        str: Path to the output rst7 file.
    """
    if n_closest < 1:
        raise ValueError("n_closest must be >= 1")

    in_dir = os.path.dirname(rst7) or "."
    base = os.path.splitext(os.path.basename(rst7))[0]
    out_rst7 = os.path.join(in_dir, f"{base}_sphere.rst7")

    lines = [
        f"parm {prmtop}",
        f"trajin {rst7}",
        "autoimage",
        "solvent :WAT",
        f"closest {n_closest} {solute_sel} noimage center closestout none outprefix sphere",
    ]
    if parmout_path:
        lines[-1] += f" parmout {parmout_path}"
    lines.append(f"trajout {out_rst7} restart")
    lines.append("run")
    lines.append("quit")

    cpptraj_input = "\n".join(lines) + "\n"
    subprocess.run([cpptraj_path], input=cpptraj_input, text=True, check=True)
    return out_rst7


def strip_xe_nobox_prmtop(prmtop_in, prmtop_out, selection=":Xe"):
    """
    Use parmed in non-interactive mode to strip a selection and remove box info.

    Args:
        prmtop_in (str): Input prmtop path.
        prmtop_out (str): Output prmtop path.
        selection (str, optional): ParmEd selection to strip (default ':Xe').
    """
    script = (
        f"parmed -p {prmtop_in} <<'EOF'\n"
        f"strip {selection} nobox\n"
        f"parmout {prmtop_out}\n"
        "quit\n"
        "EOF\n"
    )
    subprocess.run(script, shell=True, check=True, text=True)


def strip_ions_to_pdb(
    prmtop, rst7, out_pdb=None, ions=(":Cl-", ":Na+"), cpptraj_path="cpptraj"
):
    """
    Use cpptraj to strip ions from a single-frame rst7 and write a PDB.

    Args:
        prmtop (str): Path to the topology file.
        rst7 (str): Path to the single-frame restart file.
        out_pdb (str, optional): Output PDB path. Defaults to <rst7>_noions.pdb.
        ions (Sequence[str], optional): cpptraj selections for ions to strip.
        cpptraj_path (str, optional): Path to cpptraj executable.
    Returns:
        str: Path to the output PDB file.
    """
    in_dir = os.path.dirname(rst7) or "."
    base = os.path.splitext(os.path.basename(rst7))[0]
    out_pdb = out_pdb or os.path.join(in_dir, f"{base}_noions.pdb")

    lines = [
        f"parm {prmtop}",
        f"trajin {rst7}",
    ]
    for ion_sel in ions:
        lines.append(f"strip {ion_sel}")
    lines.append(f"trajout {out_pdb} pdb")
    lines.append("run")
    lines.append("quit")

    cpptraj_input = "\n".join(lines) + "\n"
    subprocess.run([cpptraj_path], input=cpptraj_input, text=True, check=True)
    return out_pdb


def add_ions_with_tleap(
    pdb_in,
    out_prmtop,
    out_rst7,
    ions,
    leaprcs=("leaprc.protein.ff14SB", "leaprc.gaff", "leaprc.water.tip3p"),
    mol2_path=None,
    frcmod_path=None,
    ligand_name="LIG",
    tleap_path="tleap",
):
    """
    Use tleap to add ions to a PDB and write prmtop/rst7.

    Args:
        pdb_in (str): Input PDB path.
        out_prmtop (str): Output prmtop path.
        out_rst7 (str): Output rst7 path.
        ions (Sequence[Tuple[str, int]]): List of (ion_name, ion_count) pairs.
        leaprcs (Sequence[str], optional): leaprc files to source.
        mol2_path (str, optional): Ligand mol2 to load.
        frcmod_path (str, optional): Ligand frcmod to load.
        ligand_name (str, optional): Ligand name for loadmol2.
        tleap_path (str, optional): Path to tleap executable.
    """
    lines = []
    for rc in leaprcs:
        lines.append(f"source {rc}")

    if mol2_path:
        lines.append(f"{ligand_name} = loadmol2 {mol2_path}")
    if frcmod_path:
        lines.append(f"loadamberparams {frcmod_path}")

    lines.append(f"prot = loadpdb {pdb_in}")
    for ion_name, ion_count in ions:
        lines.append(f"addions prot {ion_name} {int(ion_count)}")
    lines.append(f"saveamberparm prot {out_prmtop} {out_rst7}")
    lines.append("quit")

    script = "\n".join(lines) + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".in", delete=False) as handle:
        handle.write(script)
        script_path = handle.name

    try:
        subprocess.run([tleap_path, "-f", script_path], check=True, text=True)
    finally:
        os.remove(script_path)


def reionize_single_frame(
    prmtop,
    rst7,
    ions,
    out_prmtop,
    out_rst7,
    out_pdb=None,
    strip_ion_selections=(":Cl-", ":Na+"),
    leaprcs=("leaprc.protein.ff14SB", "leaprc.gaff", "leaprc.water.tip3p"),
    mol2_path=None,
    frcmod_path=None,
    ligand_name="LIG",
    cpptraj_path="cpptraj",
    tleap_path="tleap",
):
    """
    Strip existing ions with cpptraj, then re-add ions with tleap.

    Args:
        prmtop (str): Input prmtop path.
        rst7 (str): Input rst7 path.
        ions (Sequence[Tuple[str, int]]): List of (ion_name, ion_count) pairs.
        out_prmtop (str): Output prmtop path.
        out_rst7 (str): Output rst7 path.
        out_pdb (str, optional): Output PDB path from the strip step.
    Returns:
        Tuple[str, str, str]: (out_pdb, out_prmtop, out_rst7)
    """
    out_pdb = strip_ions_to_pdb(
        prmtop,
        rst7,
        out_pdb=out_pdb,
        ions=strip_ion_selections,
        cpptraj_path=cpptraj_path,
    )
    add_ions_with_tleap(
        out_pdb,
        out_prmtop,
        out_rst7,
        ions,
        leaprcs=leaprcs,
        mol2_path=mol2_path,
        frcmod_path=frcmod_path,
        ligand_name=ligand_name,
        tleap_path=tleap_path,
    )
    return out_pdb, out_prmtop, out_rst7


def generate_samples_uniform(
    topfile, trajfile, n_samples, random_seed=None, outformat="rst7", out_dir=None, verbose=False
):
    """
    Sample n_samples frames uniformly from the trajectory and save them to disk.

    Args:
        topfile (str): Path to the topology file.
        trajfile (str): Path to the trajectory file.
        n_samples (int): Number of frames to sample.
        random_seed (int, optional): Random seed for reproducibility
        outformat (str, optional): Output format for the sampled frames. Default is "rst7".
        out_dir (str, optional): Output directory for sampled frames. Defaults to CWD.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")

    traj = pt.load(trajfile, topfile)
    n_frames = traj.n_frames
    if n_samples > n_frames:
        raise ValueError(f"n_samples ({n_samples}) cannot exceed n_frames ({n_frames})")

    if n_samples == 1:
        indices = [int(round((n_frames - 1) / 2))]
    else:
        step = (n_frames - 1) / (n_samples - 1)
        indices = [int(round(i * step)) for i in range(n_samples)]

    base = os.path.splitext(os.path.basename(trajfile))[0]
    ext = outformat.lstrip(".")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    for i, frame_idx in enumerate(indices, start=1):
        filename = f"{base}_sample_{i:03d}.{ext}"
        if verbose:
            print(f"Writing frame {frame_idx} to {filename}")
        outname = os.path.join(out_dir, filename) if out_dir else filename
        pt.write_traj(outname, traj, frame_indices=[frame_idx], overwrite=True)


def get_residues_within_distance_singleframe(topfile, trajfile, target_residue, dcut):
    """
    Produce the 1-indexed list of residues within distance_cut of target_residue.
    Distance is defined as the minimum distance between any atom in target_residue and any atom in another residue.

    Args:
        topfile (str): Path to the topology file.
        trajfile (str): Path to the trajectory file.
        target_residue (int): The residue number to measure distances from (1-indexed).
        dcut (float): The distance cutoff in Angstroms.
    Returns:
        List[int]: List of 1-indexed residue numbers within distance_cut of target_residue.
    """
    if dcut < 0:
        raise ValueError("dcut must be non-negative")

    traj = pt.load(trajfile, topfile)
    top = traj.topology
    residues = top.residues

    if target_residue < 1 or target_residue > len(residues):
        raise ValueError(f"target_residue must be between 1 and {len(residues)}")

    def _atom_indices(residue):
        if hasattr(residue, "atom_indices"):
            return list(residue.atom_indices)
        return [getattr(atom, "idx", getattr(atom, "index")) for atom in residue.atoms]

    target_atom_indices = _atom_indices(residues[target_residue - 1])
    if not target_atom_indices:
        return []

    xyz = np.asarray(traj.xyz)
    if xyz.shape[0] != 1:
        raise ValueError(f"Expected a single frame, got {xyz.shape[0]}")
    xyz = xyz[0]
    target_xyz = xyz[target_atom_indices, :]
    dcut2 = float(dcut) ** 2

    residues_within = []
    for i, residue in enumerate(residues, start=1):
        if i == target_residue:
            continue
        atom_indices = _atom_indices(residue)
        if not atom_indices:
            continue

        res_xyz = xyz[atom_indices, :]
        diff = res_xyz[:, None, :] - target_xyz[None, :, :]
        dist2 = np.sum(diff * diff, axis=-1)
        if np.min(dist2) <= dcut2:
            residues_within.append(i)

    # 1-indexed residues
    return residues_within
