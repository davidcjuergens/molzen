"""Terachem input/output/submission utils"""

import os
from typing import Dict, List, Union, Optional


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
    """
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
        "Log path collisions detected, please supply manual tags to disambiguate"
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
