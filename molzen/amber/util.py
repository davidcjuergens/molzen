import os
import glob
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
    delete_input_rst7=False,
    amber_module=None,
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
        delete_input_rst7 (bool, optional): If True, delete the input rst7 after success.
        amber_module (str, optional): Module name to load before running cpptraj.
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
    if amber_module:
        cmd = f"module load {amber_module} && {cpptraj_path}"
        subprocess.run(["bash", "-lc", cmd], input=cpptraj_input, text=True, check=True)
    else:
        subprocess.run([cpptraj_path], input=cpptraj_input, text=True, check=True)
    if delete_input_rst7:
        os.remove(rst7)
    return out_rst7


def strip_xe_nobox_prmtop(
    prmtop_in,
    prmtop_out=None,
    selection=":Xe",
    parmed_path="parmed",
    amber_module=None,
    verbose=False,
    delete_prmtop_in=False,
):
    """
    Use parmed in non-interactive mode to strip a selection and remove box info.

    Args:
        prmtop_in (str): Input prmtop path.
        prmtop_out (str, optional): Output prmtop path. If None, overwrite prmtop_in.
        selection (str, optional): ParmEd selection to strip (default ':Xe').
        parmed_path (str, optional): Path to parmed executable.
        amber_module (str, optional): Module name to load before running parmed.
        verbose (bool, optional): If True, print the parmed command.
        delete_prmtop_in (bool, optional): If True, delete prmtop_in after success.
    """
    overwrite_in_place = prmtop_out is None or prmtop_out == prmtop_in
    if overwrite_in_place:
        out_dir = os.path.dirname(prmtop_in) or "."
        fd, temp_out = tempfile.mkstemp(suffix=".prmtop", dir=out_dir)
        # remove it so parmed can write to it w/o complaining that it exists
        os.close(fd)
        os.remove(temp_out)
        prmtop_out_path = temp_out
    else:
        prmtop_out_path = prmtop_out

    script = (
        f"{parmed_path} -p {prmtop_in} <<'EOF'\n"
        f"strip {selection} nobox\n"
        f"parmout {prmtop_out_path}\n"
        "go\n"
        "quit\n"
        "EOF\n"
    )
    if amber_module:
        cmd = f"module load {amber_module} && {script}"
        if verbose:
            print(f"Running: bash -lc {cmd!r}")
        subprocess.run(["bash", "-lc", cmd], check=True, text=True)
    else:
        if verbose:
            print(f"Running: {script}")
        subprocess.run(script, shell=True, check=True, text=True)
    if overwrite_in_place:
        os.replace(prmtop_out_path, prmtop_in)
    elif delete_prmtop_in:
        os.remove(prmtop_in)


def strip_ions_to_pdb(
    prmtop,
    rst7,
    out_pdb=None,
    ions=(":Cl-", ":Na+"),
    cpptraj_path="cpptraj",
    amber_module=None,
):
    """
    Use cpptraj to strip ions from a single-frame rst7 and write a PDB.

    Args:
        prmtop (str): Path to the topology file.
        rst7 (str): Path to the single-frame restart file.
        out_pdb (str, optional): Output PDB path. Defaults to <rst7>_noions.pdb.
        ions (Sequence[str], optional): cpptraj selections for ions to strip.
        cpptraj_path (str, optional): Path to cpptraj executable.
        amber_module (str, optional): Module name to load before running cpptraj.
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
    if amber_module:
        cmd = f"module load {amber_module} && {cpptraj_path}"
        subprocess.run(["bash", "-lc", cmd], input=cpptraj_input, text=True, check=True)
    else:
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
    amber_module=None,
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
        amber_module (str, optional): Module name to load before running tleap.
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
        if amber_module:
            cmd = f"module load {amber_module} && {tleap_path} -f {script_path}"
            subprocess.run(["bash", "-lc", cmd], check=True, text=True)
        else:
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
    amber_module=None,
    delete_prmtop_in=False,
    delete_rst7_in=False,
    delete_intermediate_pdb=False,
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
        amber_module=amber_module,
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
        amber_module=amber_module,
    )
    if delete_prmtop_in:
        os.remove(prmtop)
    if delete_rst7_in:
        os.remove(rst7)
    if delete_intermediate_pdb:
        os.remove(out_pdb)

    return out_pdb, out_prmtop, out_rst7


def generate_samples_uniform(
    topfile,
    trajfile,
    n_samples,
    random_seed=None,
    outformat="rst7",
    out_dir=None,
    verbose=False,
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
        if not os.path.exists(outname):
            numbered = f"{outname}.1"
            if os.path.exists(numbered):
                if verbose:
                    print(f"Renaming {numbered} -> {outname}")
                os.replace(numbered, outname)


def get_residues_within_distance_singleframe(topfile, trajfile, target_residue, dcut):
    """
    Produce the 1-indexed list of residues within distance_cut of target_residue.
    Distance is defined as the minimum distance between any atom in target_residue and any atom in another residue.

    Args:
        topfile (str): Path to the topology file.
        trajfile (str): Path to the trajectory file.
        target_residue (str): Residue mask (e.g., ':133' or ':JF4').
        dcut (float): The distance cutoff in Angstroms.
    Returns:
        List[int]: List of 1-indexed residue numbers within distance_cut of target_residue.
    """
    if dcut < 0:
        raise ValueError("dcut must be non-negative")

    traj = pt.load(trajfile, topfile)
    top = traj.topology
    residues = top.residues

    def _atom_indices(residue):
        if hasattr(residue, "atom_indices"):
            return list(residue.atom_indices)
        if hasattr(residue, "first_atom_index") and hasattr(residue, "last_atom_index"):
            return list(
                range(residue.first_atom_index, residue.last_atom_index)
            )  # not +1 because last_atom_index is exclusive
        raise AttributeError("Residue object does not expose atom indices")

    assert isinstance(target_residue, str)
    target_mask = str(target_residue)
    target_atom_indices = pt.select(target_mask, top)
    if len(target_atom_indices) == 0:
        raise ValueError(f"No atoms match selection {target_mask!r}")
    target_residue_indices = {top.atom(idx).resid + 1 for idx in target_atom_indices}
    # assert that all atoms belong to the same residue
    for idx in target_atom_indices:
        res_idx = top.atom(idx).resid + 1
        if res_idx not in target_residue_indices:
            raise ValueError(f"Selection {target_mask!r} spans multiple residues")

    xyz = np.asarray(traj.xyz)
    if xyz.shape[0] != 1:
        raise ValueError(f"Expected a single frame, got {xyz.shape[0]}")
    xyz = xyz[0]

    target_xyz = xyz[target_atom_indices, :]
    dcut2 = float(dcut) ** 2

    residues_within = []
    for i, residue in enumerate(residues, start=1):
        if i in target_residue_indices:
            continue
        atom_indices = _atom_indices(residue)
        if len(atom_indices) == 0:
            continue

        res_xyz = xyz[atom_indices, :]

        diff = res_xyz[:, None, :] - target_xyz[None, :, :]
        dist2 = np.sum(diff * diff, axis=-1)
        if np.min(dist2) <= dcut2:
            residues_within.append(i)

    # 1-indexed residues
    return residues_within


def prepare_spherical_frames(
    traj: str,
    prmtop: str,
    target_mask: str,
    n_closest_solvent: int,
    num_samples: int,
    output_dir: str,
    add_ions=[("Na+", 0), ("Cl-", 0)],
    ligand_mol2: str = None,
    ligand_frcmod: str = None,
    ligand_name: str = None,
    cpptraj_path: str = "cpptraj",
    amber_module: str = "Amber/24-CUDA-12.2.1",
    verbose: bool = False,
) -> None:
    """
    From a production Amber trajectory, randomly sample and prepare spherical capped frames.
    Usually used for preparing frames for TeraChem after running Amber MD.

    Args:
        traj (str): Path to the Amber trajectory file.
        prmtop (str): Path to the Amber prmtop file.
        target_mask (str): Residue selection mask to select sphere center (e.g., ":JF4").
        n_closest_solvent (int): Number of closest solvent molecules to retain around the target.
        num_samples (int): Number of random frames to sample uniformly from the trajectory.
        output_dir (str): Directory to save the prepared spherical capped frames.
        add_ions (list of tuples): List of tuples specifying ions to add and their counts, e.g., [("Na+", 0), ("Cl-", 0)].
                                   A count of 0 means to neutralize the system regardless of how many it takes.
        ligand_mol2 (str): Path to the ligand mol2 file.
        ligand_frcmod (str): Path to the ligand frcmod file.
        cpptraj_path (str): Path to the cpptraj executable.
        amber_module (str): Amber module to load for Amber tools.
        verbose (bool): Whether to print verbose output.
    """

    # 1. Uniformly sample N frames from the trajectory
    generate_samples_uniform(
        topfile=prmtop,
        trajfile=traj,
        n_samples=num_samples,
        out_dir=output_dir,
        verbose=verbose,
    )

    # 2. Create spherical droplets
    # TODO: probably should return the list of generated rst7s from previous step
    rst7s = glob.glob(os.path.join(output_dir, "*.rst7"))
    for rst7 in rst7s:
        make_spherical_water_droplet(
            prmtop=prmtop,
            rst7=rst7,
            solute_sel=target_mask,
            n_closest=n_closest_solvent,
            parmout_path=rst7.replace(".rst7", "_sphere.prmtop"),
            cpptraj_path=cpptraj_path,
            delete_input_rst7=True,
        )

    # 3. Remove periodic box information from prmtop files
    # TODO: probably should return the list of generated sphere prmtops from previous step
    sphere_prmtops = glob.glob(os.path.join(output_dir, "*_sphere.prmtop"))
    sphere_rst7s = [s.replace(".prmtop", ".rst7") for s in sphere_prmtops]

    for prmtop, rst7 in zip(sphere_prmtops, sphere_rst7s):
        strip_xe_nobox_prmtop(
            prmtop_in=prmtop,
            prmtop_out=None,
            amber_module=amber_module,
            verbose=verbose,
            delete_prmtop_in=True,
        )

    # 4. Remove ions remaining in the sphere,
    #    then put ions back in using tleap to once again balance charges
    nobox_prmtops = sphere_prmtops
    nobox_rst7s = [s.replace(".prmtop", ".rst7") for s in nobox_prmtops]

    for prmtop, rst7 in zip(nobox_prmtops, nobox_rst7s):
        prmtop_out = prmtop.replace("_sphere.prmtop", "_sphere_reionized.prmtop")
        rst7_out = rst7.replace("_sphere.rst7", "_sphere_reionized.rst7")
        reionize_single_frame(
            prmtop=prmtop,
            rst7=rst7,
            out_rst7=rst7_out,
            ions=add_ions,
            out_prmtop=prmtop_out,
            mol2_path=ligand_mol2,
            frcmod_path=ligand_frcmod,
            ligand_name=ligand_name,
            amber_module=amber_module,
            delete_prmtop_in=True,
            delete_rst7_in=True,
            delete_intermediate_pdb=True,
        )
