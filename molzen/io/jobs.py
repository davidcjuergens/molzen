"""Tools for executing jobs"""

import os
import stat


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
