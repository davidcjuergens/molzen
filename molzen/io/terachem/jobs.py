"""Terachem input/output/submission utils"""

import molzen.io as mzio
from molzen.io.terachem import parse as tcparse, jobs as tcjobs
from molzen.io.xyz import CoordinatesNotFoundError

import os
from typing import Dict, List, Union, Optional


def get_latest_structure(
    scrdir: str,
    xyz_storage_dir: str,
    empty_optim_xyz_ok: bool = False,
    old_input_crds: str | None = None,
) -> str:
    """Given a terachem scrdir, find the latest structure in the optimization.
    If optim.rst7 exists, simply point to that.
    If not, parse optim.xyz, grab final geometry, write to new .xyz and point to that.
    The name of the new .xyz file will be derived from the scrdir name
    Else, raise error.

    Args:
        scrdir: path to terachem scrdir
        xyz_storage_dir: directory to dump new .xyz files if needed. Must exist.
        empty_optim_xyz_ok: if True, and optim.xyz exists but contains no coordinates/frames, will point
                            to the original input geometry to restart from instead of failing to parse the optim.xyz.
                            This is to handle when TeraChem produces an optim.xyz but it's empty.
        old_input_crds: path to the original input coordinates, used if optim.xyz is empty and empty_optim_xyz_ok is True
    """

    # create paths to files that may or may not exist
    maybe_rst7 = os.path.join(scrdir, "optim.rst7")
    maybe_xyz = os.path.join(scrdir, "optim.xyz")
    scrdir_tag = os.path.basename(scrdir).replace("tc_scr.", "")
    os.makedirs(xyz_storage_dir, exist_ok=True)

    # if optim.rst7 exists, symlink that and point to it in new job
    if os.path.exists(maybe_rst7):
        new_rst7_path = os.path.join(xyz_storage_dir, f"{scrdir_tag}.rst7")
        os.symlink(maybe_rst7, new_rst7_path)
        return new_rst7_path

    # if optim.xyz exists, parse it, grab final frame, write to new .xyz and point to that in new job
    elif os.path.exists(maybe_xyz):
        try:
            mol = mzio.Molecule.from_xyz(maybe_xyz)
            final_frame = mol[-1]
            
        except CoordinatesNotFoundError:
        # Sometimes, terachem produces optim.xyz without any data.
        # In this case, if user passed in empty_optim_xyz_ok=True, 
        # point to the original input geometry instead of raising.
            if empty_optim_xyz_ok and old_input_crds is not None:
                return old_input_crds
            else:
                raise
        new_xyz_path = os.path.join(xyz_storage_dir, f"{scrdir_tag}.xyz")
        assert not os.path.exists(new_xyz_path), (
            f".xyz file {new_xyz_path} already exists."
        )
        final_frame.to_xyz(new_xyz_path)
        return new_xyz_path

    else:
        raise FileNotFoundError(f"No optim.rst7 or optim.xyz found in {scrdir}")


def restart_terachem_optimization_from_latest(
    tc_stdouts: list,
    new_workdir: str,
    terachem_exe: str,
    clobber: bool = False,
    kwarg_updates: dict = None,
    empty_optim_xyz_ok: bool = False,
):
    """Create jobs that restart optimization from latest geometry in scrdir

    Args:
        tc_stdouts: list of paths terachem output files
        new_workdir: new directory to dump jobs
        terachem_exe: terachem executable
        clobber: whether to overwrite existing files
        kwarg_updates: dict of any additional keyword arguments to place into new jobs
                       (e.g. if you want to change method, basis, etc. for the restart)
        empty_optim_xyz_ok: if True, and optim.xyz exists but contains no coordinates/frames, will point
                            to the original input geometry to restart from instead of failing to parse the optim.xyz.
    """
    parsed = [tcparse.parse_terachem_output(s) for s in tc_stdouts]

    #### Process job args ####
    # get args from stdout
    job_args = [p["input_args"] for p in parsed]

    # update job args with any additional keyword arguments
    if kwarg_updates is not None:
        for args in job_args:
            args.update(kwarg_updates)

    # extract key items that must be removed or passed back in
    # pop(..., None) because we may not have constraints and make_terachem_input can deal with None constraints
    job_constraint_lists = [
        job_args[i].pop("constraints", None) for i in range(len(job_args))
    ]
    old_scrdirs = [k.pop("scrdir") for k in job_args]
    # pop old coordinates
    old_coordinates = [k.pop("coordinates") for k in job_args]

    # get final structures -- assume either optim.rst7 or optim.xyz
    # if optim.rst7 exists, use that.
    # if optim.xyz, parse it and grab final geometry from that.
    latest_geometries = []
    for i, d in enumerate(old_scrdirs):
        xyz_storage_dir = os.path.join(new_workdir, "latest_xyzs")
        old_crds = old_coordinates[i]
        latest = get_latest_structure(
            d,
            xyz_storage_dir,
            empty_optim_xyz_ok=empty_optim_xyz_ok,
            old_input_crds=old_crds,
        )
        latest_geometries.append(latest)

    #####
    # now make new jobs with the same args but new coordinates and new workdir
    #####
    tcjobs.make_terachem_job_array(
        latest_geometries,
        workdir=new_workdir,
        tc_kwargs=job_args,
        terachem_exe=terachem_exe,
        clobber=clobber,
        tags=None,
        constraint_lists=job_constraint_lists,
    )


def make_terachem_input(
    xyz_path: str,
    tc_kwargs: dict,
    input_writedir: str,
    workdir: str,
    tag=None,
    clobber=False,
    constraints: Optional[List[str]] = None,
    scrdir: Optional[str] = None,
):
    """Makes a single terachem input file

    Args:
        xyz_path: path to xyz input file to operate on
        tc_kwargs: dictionary of terachem keywords arguments
        input_writedir: directory to write the terachem input file
        workdir: working directory for terachem to run in
        outdir: directory to write
        tag: optional tag to append to scrdir and input file name
        clobber: whether to overwrite existing input file
        constraints: optional list of strings representing constraints to add. Each entry is one constraint line. The lines will be surrounded by $constraints ... $end

    """

    # make scr dir
    suffixes_to_remove = (".xyz", ".rst7")
    prefix = os.path.basename(xyz_path)
    for suffix in suffixes_to_remove:
        if prefix.endswith(suffix):
            prefix = prefix[: -len(suffix)]
            break

    if tag is not None:
        prefix = f"{prefix}_{tag}"

    # ensure there is no capitalization in the prefix, since terachem silently converts to lowercase...
    if any([c.isupper() for c in prefix]):
        print(
            f"WARNING: prefix {prefix} contains uppercase letters. Converting to lowercase"
        )
        prefix = prefix.lower()

    # make sure we didn't accidentally think it was going to be some other scrdir
    assert tc_kwargs.get("scrdir") is None
    assert tc_kwargs.get("coordinates") is None

    if scrdir is None:
        job_scrdir = os.path.join(workdir, f"tc_scr.{prefix}")
    else:
        job_scrdir = os.path.join(scrdir, f"tc_scr.{prefix}")

    tc_kwargs["scrdir"] = job_scrdir
    tc_kwargs["coordinates"] = xyz_path

    ## method-specific things
    # ccbox
    if tc_kwargs.get("ccbox", None) == "yes":
        tc_kwargs["ccbox_scratch_dir"] = job_scrdir

    # make input file
    longest_key = max([len(key) for key in tc_kwargs.keys()])
    spacer = longest_key + 5
    input_path = os.path.join(input_writedir, f"{prefix}.in")

    if not clobber:
        assert not os.path.exists(input_path), (
            f"Refusing to clobber already existing {input_path}"
        )

    with open(input_path, "w") as f:
        for key, val in tc_kwargs.items():
            space = " " * (spacer - len(key))
            f.write(f"{key}{space}{val}\n")

        if constraints is not None and len(constraints):
            f.write("\n")
            f.write("$constraints\n")
            for constraint_line in constraints:
                f.write(f"{constraint_line}\n")
            f.write("$end\n")
        f.write("end\n")

    return input_path


def make_terachem_job_array(
    xyzs: list,
    workdir: str,
    tc_kwargs: Union[List[Dict], Dict],
    tags: list = None,
    terachem_exe: str = "terachem",
    clobber: bool = False,
    task_filename: str = "terachem_tasks.txt",
    constraint_lists: Optional[List[List[str]]] = None,
    scrdir: Optional[str] = None,
    make_scrdir: bool = True,
):
    """Make a batch of terachem jobs that can be executed

    Args:
        xyzs: list of .xyz files to perform computations on
        workdir: directory where job-submission files are written and default terachem scrdir location
        tc_kwargs: either a list of dictionaries, or a single dict.
                    If it's a single dict, will duplicate kwargs across xyzs
        tags: optional list of tags to attach to scr dirs and output files
        terachem_exe: terachem executable command
        clobber: whether to overwrite existing files
        constraint_lists: optional list of constraint lists. if tc_kwargs is a dict, this should be only one list of constraints.
                          If tc_kwargs is a list of dicts, this should be a list of lists of constraints, one per job.
        scrdir: optional path for dumping terachem scratch dirs and job log files. If not provided, will be workdir.
        make_scrdir: have this function create the base scrdir
    """
    if scrdir is not None and make_scrdir:
        os.makedirs(scrdir, exist_ok=False)

    if constraint_lists is not None:
        if isinstance(tc_kwargs, list):
            assert len(tc_kwargs) == len(constraint_lists) and isinstance(
                constraint_lists, list
            )
            assert isinstance(constraint_lists[0], list)
        else:
            assert isinstance(constraint_lists, list) and isinstance(
                constraint_lists[0], str
            )
            constraint_lists = [constraint_lists for _ in xyzs]
    else:
        pass

    if isinstance(tc_kwargs, list):
        assert len(tc_kwargs) == len(xyzs)
        tc_kwargs_list = tc_kwargs
    else:
        tc_kwargs_list = [tc_kwargs.copy() for _ in xyzs]

    tc_input_dir = os.path.join(workdir, "tc_inputs/")
    os.makedirs(tc_input_dir, exist_ok=True)
    one_liners = []

    log_paths = []
    for i, xyz in enumerate(xyzs):
        tag = None
        if tags is not None:
            tag = tags[i]

        # make a terachem input file
        input_path = make_terachem_input(
            xyz_path=xyz,
            tc_kwargs=tc_kwargs_list[i],
            workdir=workdir,
            input_writedir=tc_input_dir,
            tag=tag,
            clobber=clobber,
            constraints=constraint_lists[i] if constraint_lists is not None else None,
            scrdir=scrdir,
        )

        if scrdir is not None:
            logdir = scrdir
        else:
            logdir = workdir
        log_path = os.path.join(
            logdir, f"stdout_{os.path.basename(input_path).replace('.in', '.log')}"
        )

        # make one liner to run terachem for this input
        one_liner = f"{terachem_exe} {input_path} > {log_path} 2>&1"
        one_liners.append(one_liner)

        # track log paths to ensure we don't have any collisions
        log_paths.append(log_path)

    # ensure no log path collisions
    assert len(log_paths) == len(set(log_paths)), (
        "Log path collisions detected, please supply manual tags to disambiguate:\n"
        + "\n".join(log_paths)
    )

    # write out jobs to a task array
    tasks_file = os.path.join(workdir, task_filename)
    if not clobber:
        assert not os.path.exists(tasks_file), (
            f"Refusing to clobber already existing {tasks_file}"
        )

    with open(tasks_file, "w") as f:
        for line in one_liners:
            f.write(f"{line}\n")

    print(f"Wrote {len(one_liners)} terachem jobs to {tasks_file}")
