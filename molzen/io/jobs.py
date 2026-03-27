"""Tools for executing jobs"""

import os
import stat
from pathlib import Path


def generate_parallel_launcher(
    cuda_indices,
    jobs_per_gpu,
    job_file,
    output_script="run_terachem.sh",
    gpus_per_job=1,
):
    """
    Generates a shell script that uses GNU Parallel to run jobs on specific GPUs.
    NOTE: You have to have GNU Parallel installed and available in your PATH for this to work.

    Args:
        cuda_indices (list): List of integers, e.g., [0, 1, 2, 3] or [0, 2]
        jobs_per_gpu (int): How many jobs to run at once per GPU.
        job_file (str): Path to the text file containing the jobs.
        output_script (str): Name of the shell script to create.
        gpus_per_job (int): Number of GPUs to assign per job.
    """

    if not cuda_indices:
        raise ValueError("cuda_indices cannot be empty.")
    if jobs_per_gpu < 1:
        raise ValueError("jobs_per_gpu must be >= 1.")
    if gpus_per_job < 1:
        raise ValueError("gpus_per_job must be >= 1.")

    # 1. Calculate total parallel slots (concurrency)
    num_gpus = len(cuda_indices)
    available_gpu_slots = num_gpus * jobs_per_gpu
    total_slots = available_gpu_slots // gpus_per_job
    if total_slots < 1:
        raise ValueError(
            "Not enough total GPU slots for requested gpus_per_job. "
            f"Got {num_gpus} GPUs, jobs_per_gpu={jobs_per_gpu}, "
            f"gpus_per_job={gpus_per_job}."
        )

    # 2. Format the GPU list for Bash
    # We turn the list [0, 2] into a bash array string "0 2"
    gpu_list_str = " ".join(map(str, cuda_indices))
    expanded_gpu_slots = [str(gpu) for _ in range(jobs_per_gpu) for gpu in cuda_indices]
    gpu_groups = [
        ",".join(expanded_gpu_slots[i : i + gpus_per_job])
        for i in range(0, total_slots * gpus_per_job, gpus_per_job)
    ]
    gpu_groups_str = " ".join([f'"{group}"' for group in gpu_groups])

    # 3. Construct the script content
    # We map each Parallel Slot ID {%} to a precomputed GPU group.
    # Logic:
    #   We define the GPU group array inside the command.
    #   Slot {%} (1..N) is converted to a 0-based index.
    #   That index selects the comma-separated CUDA_VISIBLE_DEVICES value.
    script_content = f"""#!/bin/bash

# --- Configuration ---
# Job File: {job_file}
# GPUs to use: {cuda_indices}
# Jobs per GPU: {jobs_per_gpu}
# GPUs per job: {gpus_per_job}
# Total Concurrency: {total_slots}
# ---------------------

echo "Starting {total_slots} parallel workers on GPUs: {gpu_list_str}..."

# We set SHELL to bash to ensure array syntax works inside parallel
export SHELL=/bin/bash

# Explain the command:
# --jobs {total_slots} : Number of workers total
# --joblog run.log     : Keeps a record of finished jobs (safe to Ctrl+C and resume)
# --resume             : Skips jobs listed as 'finished' in run.log
# {{%}}                : The 'Slot ID' (1 to {total_slots})
# {{}}                 : The job command from the text file
# Each slot gets one static CUDA_VISIBLE_DEVICES assignment.

parallel --jobs {total_slots} --joblog run.log --resume \\
    'gpu_groups=({gpu_groups_str}); \\
     slot=$(({{%}} - 1)); \\
     idx=$((slot % {total_slots})); \\
     device=${{gpu_groups[$idx]}}; \\
     export CUDA_VISIBLE_DEVICES=$device; \\
     echo "Slot {{%}} running on GPUs $device: {{}}"; \\
     eval {{}}' \\
    :::: {job_file}

echo "All jobs finished."
"""

    # 4. Write the file
    with open(output_script, "w") as f:
        f.write(script_content)

    # 5. Make it executable
    st = os.stat(output_script)
    os.chmod(output_script, st.st_mode | stat.S_IEXEC)

    print(f"Success! Generated '{output_script}'. Run it with: ./{output_script}")


def generate_parallel_srun_launcher(
    job_file,
    output_script="run_jobs.sh",
    cpus_per_job=16,
    gpus_per_job=1,
    jobs_per_gpu=1,
    jobs_in_flight=None,
    joblog="parallel_joblog.txt",
    resume=True,
    srun_extra_args=None,
    bash_setup=None,
):
    """
    Generate a shell script that uses GNU Parallel + srun to run one command per
    line from a job file across a Slurm allocation.

    This is intended for multi-node Slurm jobs (e.g. Perlmutter), where GNU
    Parallel feeds commands and each command is launched as its own `srun` step.

    Parameters
    ----------
    job_file : str or Path
        Path to a text file containing one shell command per line.
    output_script : str or Path, default="run_jobs.sh"
        Name of the shell script to create.
    cpus_per_job : int, default=16
        CPUs to request for each launched job step.
    gpus_per_job : int, default=1
        Number of GPUs each launched command should see via
        `CUDA_VISIBLE_DEVICES`.
    jobs_per_gpu : int, default=1
        How many concurrent commands to allow per physical GPU. A value of 2
        means the launcher will map two GNU Parallel worker slots onto each GPU.
    jobs_in_flight : int or None, default=None
        Max number of jobs GNU Parallel should keep active simultaneously.
        If None, the generated script infers this from the Slurm allocation as

            total_gpu_slots_in_allocation // gpus_per_job

        where `total_gpu_slots_in_allocation = total_gpus_in_allocation *
        jobs_per_gpu`. If `SLURM_CPUS_ON_NODE` is available, the generated
        script also caps concurrency so it does not exceed the CPU allocation.
    joblog : str, default="parallel_joblog.txt"
        GNU Parallel joblog filename.
    resume : bool, default=True
        If True, use GNU Parallel `--resume-failed`, so previously successful
        jobs are skipped while failed jobs are retried.
    srun_extra_args : list[str] or str or None, default=None
        Extra arguments appended to the srun command, e.g.
        ["--cpu-bind=cores"] or "--cpu-bind=cores".
        Note: `--exact` is already included by default in the generated script.
    bash_setup : str or None, default=None
        Optional shell snippet inserted before each command runs. Useful for
        module loads, `source activate`, `cd`, environment exports, etc.

        Example:
            bash_setup=\"\"\"
            module use /global/cfs/cdirs/m5151/software/module
            module load TeraChem/tc250307.lua
            cd $SLURM_SUBMIT_DIR
            \"\"\"

    Notes
    -----
    - This script is meant to be executed *inside* a Slurm allocation
      (sbatch or salloc).
    - Each line of `job_file` should be a complete shell command.
    - Blank lines and comment lines starting with `#` are ignored.
    - GPU selection is handled by setting `CUDA_VISIBLE_DEVICES` inside each
      launched `srun` step. To make `jobs_per_gpu > 1` possible, the generated
      script launches overlapping Slurm steps on each node and requests access
      to that node's full GPU set, then constrains each command with
      `CUDA_VISIBLE_DEVICES`.
    - This function does not itself submit a Slurm job; it only generates the
      launcher script.
    """
    job_file = Path(job_file)
    output_script = Path(output_script)

    if not job_file.exists():
        raise FileNotFoundError(f"job_file does not exist: {job_file}")
    if cpus_per_job < 1:
        raise ValueError("cpus_per_job must be >= 1")
    if gpus_per_job < 1:
        raise ValueError("gpus_per_job must be >= 1")
    if jobs_per_gpu < 1:
        raise ValueError("jobs_per_gpu must be >= 1")
    if jobs_in_flight is not None and jobs_in_flight < 1:
        raise ValueError("jobs_in_flight must be >= 1 when provided")

    if srun_extra_args is None:
        srun_extra_args_str = ""
    elif isinstance(srun_extra_args, str):
        srun_extra_args_str = srun_extra_args.strip()
    else:
        srun_extra_args_str = " ".join(str(x) for x in srun_extra_args).strip()

    resume_flag = "--resume-failed" if resume else ""
    bash_setup = (bash_setup or "").rstrip()

    # Preserve literal text for the shell script.
    if jobs_in_flight is None:
        jobs_assignment = 'JOBS_IN_FLIGHT=""'
    else:
        jobs_assignment = f'JOBS_IN_FLIGHT="{jobs_in_flight}"'

    setup_block = ""
    if bash_setup:
        setup_block = f"""
# User-provided environment/setup snippet run before each job command.
read -r -d '' BASH_SETUP <<'EOF_SETUP' || true
{bash_setup}
EOF_SETUP
"""

    script = f"""#!/bin/bash
set -euo pipefail

JOB_FILE="{job_file}"
CPUS_PER_JOB="{cpus_per_job}"
GPUS_PER_JOB="{gpus_per_job}"
JOBS_PER_GPU="{jobs_per_gpu}"
JOBLOG="{joblog}"
{jobs_assignment}
SRUN_EXTRA_ARGS='{srun_extra_args_str}'

{setup_block}
if [[ ! -f "$JOB_FILE" ]]; then
    echo "Error: job file not found: $JOB_FILE" >&2
    exit 1
fi

if ! command -v parallel >/dev/null 2>&1; then
    echo "Error: GNU Parallel is not available in PATH." >&2
    exit 1
fi

if ! command -v srun >/dev/null 2>&1; then
    echo "Error: srun is not available in PATH." >&2
    exit 1
fi

if ! command -v scontrol >/dev/null 2>&1; then
    echo "Error: scontrol is not available in PATH." >&2
    exit 1
fi

if [[ ! "$GPUS_PER_JOB" =~ ^[0-9]+$ ]] || [[ ! "$JOBS_PER_GPU" =~ ^[0-9]+$ ]]; then
    echo "Error: GPUS_PER_JOB and JOBS_PER_GPU must be integers." >&2
    exit 1
fi

# Infer concurrency from the active Slurm allocation if not explicitly set.
if [[ -z "${{SLURM_JOB_NODELIST:-}}" ]]; then
    echo "Error: SLURM_JOB_NODELIST is not set." >&2
    echo "Run this script inside an sbatch/salloc allocation." >&2
    exit 1
fi

GPUS_PER_NODE="${{SLURM_GPUS_ON_NODE:-}}"
if [[ -z "$GPUS_PER_NODE" ]]; then
    echo "Error: SLURM_GPUS_ON_NODE is not set, so I cannot infer GPU capacity." >&2
    exit 1
fi
if [[ ! "$GPUS_PER_NODE" =~ ^[0-9]+$ ]]; then
    echo "Error: SLURM_GPUS_ON_NODE must be an integer for this launcher." >&2
    echo "  Got: $GPUS_PER_NODE" >&2
    exit 1
fi

mapfile -t ALLOC_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [[ "${{#ALLOC_NODES[@]}}" -eq 0 ]]; then
    echo "Error: could not expand SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST" >&2
    exit 1
fi

SLOT_ASSIGNMENTS_FILE=$(mktemp)
FILTERED_JOB_FILE=$(mktemp)
trap 'rm -f "$SLOT_ASSIGNMENTS_FILE" "$FILTERED_JOB_FILE"' EXIT

for node in "${{ALLOC_NODES[@]}}"; do
    expanded_gpu_slots=()
    for ((rep = 0; rep < JOBS_PER_GPU; rep++)); do
        for ((gpu = 0; gpu < GPUS_PER_NODE; gpu++)); do
            expanded_gpu_slots+=("$gpu")
        done
    done

    slots_on_node=$(( ${{#expanded_gpu_slots[@]}} / GPUS_PER_JOB ))
    for ((slot_idx = 0; slot_idx < slots_on_node; slot_idx++)); do
        start=$(( slot_idx * GPUS_PER_JOB ))
        device_group="${{expanded_gpu_slots[$start]}}"
        for ((offset = 1; offset < GPUS_PER_JOB; offset++)); do
            device_group+=",${{expanded_gpu_slots[$((start + offset))]}}"
        done
        printf '%s|%s\\n' "$node" "$device_group" >> "$SLOT_ASSIGNMENTS_FILE"
    done
done

MAX_JOBS_IN_FLIGHT=$(wc -l < "$SLOT_ASSIGNMENTS_FILE")
if [[ "$MAX_JOBS_IN_FLIGHT" -lt 1 ]]; then
    echo "Error: Computed MAX_JOBS_IN_FLIGHT < 1." >&2
    echo "  nodes=${{#ALLOC_NODES[@]}}" >&2
    echo "  SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE" >&2
    echo "  JOBS_PER_GPU=$JOBS_PER_GPU" >&2
    echo "  GPUS_PER_JOB=$GPUS_PER_JOB" >&2
    exit 1
fi

CPU_LIMIT_MSG="not evaluated"
if [[ -n "${{SLURM_CPUS_ON_NODE:-}}" ]] && [[ "${{SLURM_CPUS_ON_NODE}}" =~ ^[0-9]+$ ]]; then
    MAX_CPU_JOBS=$(( (${{#ALLOC_NODES[@]}} * SLURM_CPUS_ON_NODE) / CPUS_PER_JOB ))
    if [[ "$MAX_CPU_JOBS" -lt 1 ]]; then
        echo "Error: Computed CPU-based concurrency < 1." >&2
        echo "  nodes=${{#ALLOC_NODES[@]}}" >&2
        echo "  SLURM_CPUS_ON_NODE=$SLURM_CPUS_ON_NODE" >&2
        echo "  CPUS_PER_JOB=$CPUS_PER_JOB" >&2
        exit 1
    fi
    if [[ "$MAX_CPU_JOBS" -lt "$MAX_JOBS_IN_FLIGHT" ]]; then
        MAX_JOBS_IN_FLIGHT="$MAX_CPU_JOBS"
    fi
    CPU_LIMIT_MSG="$MAX_CPU_JOBS"
elif [[ -n "${{SLURM_CPUS_ON_NODE:-}}" ]]; then
    CPU_LIMIT_MSG="skipped (non-integer: $SLURM_CPUS_ON_NODE)"
fi

if [[ -z "${{JOBS_IN_FLIGHT}}" ]]; then
    JOBS_IN_FLIGHT="$MAX_JOBS_IN_FLIGHT"
elif [[ "$JOBS_IN_FLIGHT" -gt "$MAX_JOBS_IN_FLIGHT" ]]; then
    echo "Error: requested JOBS_IN_FLIGHT exceeds allocation-backed slot capacity." >&2
    echo "  JOBS_IN_FLIGHT=$JOBS_IN_FLIGHT" >&2
    echo "  MAX_JOBS_IN_FLIGHT=$MAX_JOBS_IN_FLIGHT" >&2
    exit 1
fi

echo "Job file         : $JOB_FILE"
echo "Output script    : $0"
echo "CPUs per job     : $CPUS_PER_JOB"
echo "GPUs per job     : $GPUS_PER_JOB"
echo "Jobs per GPU     : $JOBS_PER_GPU"
echo "Jobs in flight   : $JOBS_IN_FLIGHT"
echo "Max jobs/alloc   : $MAX_JOBS_IN_FLIGHT"
echo "Joblog           : $JOBLOG"
echo "Slurm job ID     : ${{SLURM_JOB_ID:-<none>}}"
echo "Slurm nodes      : ${{#ALLOC_NODES[@]}}"
echo "Slurm GPUs/node  : ${{SLURM_GPUS_ON_NODE:-<unknown>}}"
echo "Slurm CPUs/node  : ${{SLURM_CPUS_ON_NODE:-<unknown>}}"
echo "CPU slot limit   : $CPU_LIMIT_MSG"

export SHELL=/bin/bash
export SLOT_ASSIGNMENTS_FILE
export SRUN_EXTRA_ARGS
export BASH_SETUP

# Filter out blank lines and full-line comments before feeding into GNU Parallel.
# Each surviving line is treated as a complete shell command.
# GNU Parallel slot IDs are used as stable worker IDs, and each worker is bound
# to a precomputed {{node, CUDA_VISIBLE_DEVICES}} pair from SLOT_ASSIGNMENTS_FILE.
grep -Ev '^[[:space:]]*($|#)' "$JOB_FILE" > "$FILTERED_JOB_FILE"
NUM_COMMANDS=$(wc -l < "$FILTERED_JOB_FILE")
echo "Commands found   : $NUM_COMMANDS"

if [[ "$NUM_COMMANDS" -lt 1 ]]; then
    echo "Error: no runnable commands found in $JOB_FILE." >&2
    exit 1
fi

parallel --jobs "$JOBS_IN_FLIGHT" --joblog "$JOBLOG" {resume_flag} --line-buffer \\
    '
    slot_id={{%}}
    cmd={{}}
    assignment=$(sed -n "${{slot_id}}p" "$SLOT_ASSIGNMENTS_FILE")
    if [[ -z "$assignment" ]]; then
        echo "[parallel slot {{%}}] no slot assignment found" >&2
        exit 1
    fi
    node=${{assignment%%|*}}
    device=${{assignment#*|}}
    if [[ -z "$node" || -z "$device" || "$node" == "$assignment" ]]; then
        echo "[parallel slot {{%}}] malformed slot assignment: $assignment" >&2
        exit 1
    fi

    echo "[parallel slot {{%}}] launching on $(date) node=$node cuda=$device: $cmd"

    srun --overlap --exact -N 1 -n 1 -w "$node" -c '"$CPUS_PER_JOB"' --gres=gpu:'"$GPUS_PER_NODE"' $SRUN_EXTRA_ARGS \\
        bash -lc "
            set -euo pipefail
            {"$BASH_SETUP" if bash_setup else ":"}
            export CUDA_VISIBLE_DEVICES=$device
            echo \\"[srun host=\$(hostname) procid=\${{SLURM_PROCID:-NA}} localid=\${{SLURM_LOCALID:-NA}} cuda=\${{CUDA_VISIBLE_DEVICES:-unset}}]\\"
            $cmd
        "
    ' \\
    :::: "$FILTERED_JOB_FILE"

echo "All jobs finished."
"""

    with open(output_script, "w", encoding="utf-8") as f:
        f.write(script)

    st = os.stat(output_script)
    os.chmod(output_script, st.st_mode | stat.S_IEXEC)

    print(f"Success! Generated '{output_script}'.")
    if output_script.is_absolute():
        print(f"Run it inside your Slurm allocation with: {output_script}")
    else:
        print(f"Run it inside your Slurm allocation with: ./{output_script}")
