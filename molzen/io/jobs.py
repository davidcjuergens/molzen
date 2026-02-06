"""Tools for executing jobs"""

import os
import stat


def generate_parallel_launcher(
    cuda_indices, jobs_per_gpu, job_file, output_script="run_terachem.sh"
):
    """
    Generates a shell script that uses GNU Parallel to run jobs on specific GPUs.
    NOTE: You have to have GNU Parallel installed and available in your PATH for this to work.

    Args:
        cuda_indices (list): List of integers, e.g., [0, 1, 2, 3] or [0, 2]
        jobs_per_gpu (int): How many jobs to run at once per GPU.
        job_file (str): Path to the text file containing the jobs.
        output_script (str): Name of the shell script to create.
    """

    # 1. Calculate total parallel slots (concurrency)
    num_gpus = len(cuda_indices)
    total_slots = num_gpus * jobs_per_gpu

    # 2. Format the GPU list for Bash
    # We turn the list [0, 2] into a bash array string "0 2"
    gpu_list_str = " ".join(map(str, cuda_indices))

    # 3. Construct the script content
    # We use a specific logic to map the Parallel Slot ID {%} to a GPU index
    # Logic:
    #   We define the GPU array inside the command.
    #   We take the slot number {%} (which goes 1..N), subtract 1, and modulo by num_gpus.
    #   We use that result to index into our GPU array.
    script_content = f"""#!/bin/bash

# --- Configuration ---
# Job File: {job_file}
# GPUs to use: {cuda_indices}
# Jobs per GPU: {jobs_per_gpu}
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

parallel --jobs {total_slots} --joblog run.log --resume \\
    'gpus=({gpu_list_str}); \\
     slot=$(({{%}} - 1)); \\
     idx=$((slot % {num_gpus})); \\
     device=${{gpus[$idx]}}; \\
     export CUDA_VISIBLE_DEVICES=$device; \\
     echo "Slot {{%}} running on GPU $device: {{}}"; \\
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
