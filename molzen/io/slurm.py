"""Slurm related tools"""

import os
import subprocess
import shutil
import time


def count_squeue_jobs(lines):
    """Counts number of jobs in an squeue output, accounting for array jobs."""
    job_count = 0
    for line in lines:
        # Check if its an array submission
        if "[" in line and "]" in line:
            task_range = line.split()[0].split("_")[1].strip("[]").split("-")
            end_idx = int(task_range[1])
            start_idx = int(task_range[0])
            job_count += end_idx - start_idx + 1
        else:
            # just a single job
            job_count += 1
    return job_count


def get_current_job_count():
    """Get current number of running and pending jobs for user in SLURM queue."""
    # Get the current user's username
    user = os.environ.get("USER")
    if not user:
        print("Error: Could not determine username from $USER environment variable.")
        return -1

    # Command to get all Running (R) and Pending (PD) jobs for the user
    # -h removes the header
    # -t R,PD filters for Running and Pending states
    # -u $USER filters for the current user
    cmd_running = ["squeue", "-h", "-t", "R", "-u", user]
    cmd_pending = ["squeue", "-h", "-t", "PD", "-u", user]

    # Run the command and capture output
    result_running = subprocess.run(
        cmd_running, capture_output=True, text=True, check=True
    )
    result_pending = subprocess.run(
        cmd_pending, capture_output=True, text=True, check=True
    )

    # Count the number of lines in the output. Each line is a job.
    lines_running = result_running.stdout.strip().splitlines()
    lines_pending = result_pending.stdout.strip().splitlines()

    running = count_squeue_jobs(lines_running)
    pending = count_squeue_jobs(lines_pending)
    total = running + pending

    return running, pending, total


def slurm_bot(
    task_file: str,
    slurm_kwargs: dict,
    max_queued: int = 15,
    sleep_time_minutes: int = 5,
    use_wrap: bool = False,
):
    """Submits jobs from a task file to SLURM queue, ensuring that the total number of queued jobs does not exceed max_queued.

    Args:
        task_file (str): Path to the file containing tasks to submit.
        slurm_kwargs (dict): Dictionary of SLURM submission parameters.
        max_queued (int): Maximum number of jobs allowed in the queue.
        sleep_time_minutes (int): Minutes to sleep before rechecking the queue.
        use_wrap (bool): Whether to use --wrap for job submission. if not, creates individual sbatch scripts.
    """
    # make a copy of the original task file for safety
    remaining_tasks_file = task_file + ".remaining"
    failed_submissions_file = task_file + ".submission_failed"

    if not os.path.exists(remaining_tasks_file):
        print(f"Copying {task_file} to {remaining_tasks_file} to start.")
        shutil.copy(task_file, remaining_tasks_file)
    else:
        print(f"Resuming from {remaining_tasks_file}.")

    while True:
        # We read the file inside the loop, in case it's modified externally
        try:
            with open(remaining_tasks_file, "r") as f:
                tasks = f.readlines()
        except FileNotFoundError:
            print(f"Task file {remaining_tasks_file} not found. Bot is exiting.")
            break

        if not tasks:
            print("All tasks submitted. Bot is exiting.")
            break

        running, pending, total = get_current_job_count()
        print(f"Current jobs - Running: {running}, Pending: {pending}, Total: {total}")

        if total < max_queued:
            n_to_submit = max_queued - total
            print(f"Queue has {n_to_submit} open slots. Attempting to submit...")

            submitted_this_round = 0
            tasks_processed_this_round = []

            # submit up to n_to_submit jobs
            while submitted_this_round < n_to_submit and tasks:
                # Take the first task off the list to process it
                task_line = tasks.pop(0)
                tasks_processed_this_round.append(task_line)

                task = task_line.strip()
                if not task:
                    # It was just an empty line, don't count it
                    continue

                # Prepare SLURM submission command

                if use_wrap:
                    cmd = ["sbatch"]
                    for key, value in slurm_kwargs.items():
                        if key.startswith("--"):
                            cmd.append(f"{key}={value}")
                        elif key.startswith("-"):
                            cmd.append(f"{key} {value}")
                        else:
                            raise ValueError(f"Invalid SLURM argument key: {key}")
                    cmd.append(f"--wrap='{task}'")

                else:
                    # make a temporary sbatch script
                    sbatch_script = "#!/bin/bash\n"
                    for key, value in slurm_kwargs.items():
                        if key.startswith("--"):
                            sbatch_script += f"#SBATCH {key}={value}\n"
                        elif key.startswith("-"):
                            sbatch_script += f"#SBATCH {key} {value}\n"
                        else:
                            raise ValueError(f"Invalid SLURM argument key: {key}")
                    sbatch_script += f"\n{task}\n"
                    # write to a temp file
                    temp_script_path = "temp_sbatch_script.sh"
                    with open(temp_script_path, "w") as temp_f:
                        temp_f.write(sbatch_script)
                    cmd = ["sbatch", temp_script_path]
                print(f"Submitting: {' '.join(cmd)}")

                # --- Submit the job ---
                result = subprocess.run(cmd, capture_output=True, text=True)

                if not use_wrap:
                    # remove the temporary script
                    os.remove(temp_script_path)

                if result.returncode == 0:
                    # --- SUCCESS ---
                    print(f"  -> SUCCESS: {result.stdout.strip()}")
                    submitted_this_round += 1
                else:
                    # --- FAILURE ---
                    error_msg = result.stderr.strip()
                    print(f"  -> FAILURE: {error_msg}")

                    # Log the failed submission
                    with open(failed_submissions_file, "a") as fail_f:
                        fail_f.write(f"{task_line.strip()}  # Error: {error_msg}\n")

            print(
                f"Submitted {submitted_this_round} jobs this round. {len(tasks)} tasks remaining."
            )

            # After processing, write back the remaining tasks
            with open(remaining_tasks_file, "w") as f:
                f.writelines(tasks)

        else:
            print(
                f"Max queued jobs reached ({max_queued}). Sleeping for {sleep_time_minutes} minutes."
            )
            time.sleep(sleep_time_minutes * 60)

def create_slurm_array_script(task_file, sbatch_params, output_script="submit.sh"):
    """
    Written by Gemini 3.0
    Generates a Slurm array script based on a file of commands.
    
    Args:
        task_file (str): Path to the file containing one command per line.
        sbatch_params (dict): Dictionary of sbatch flags (keys) and values.
        output_script (str): Name of the output .sh file.
    """
    # 1. Get absolute path of task file so the script runs from anywhere
    abs_task_file = os.path.abspath(task_file)
    
    # 2. Count lines to determine array size
    with open(abs_task_file, 'r') as f:
        num_tasks = sum(1 for line in f if line.strip())

    with open(output_script, 'w') as f:
        f.write("#!/bin/bash\n")
        
        # 3. Write User SBATCH params
        for flag, value in sbatch_params.items():
            # Handles both long (--time=...) and short (-p ...) formatting styles
            separator = "=" if flag.startswith("--") else " "
            f.write(f"#SBATCH {flag}{separator}{value}\n")
            
        # 4. Write Array Directive (1 to N)
        f.write(f"#SBATCH --array=1-{num_tasks}\n\n")
        
        # 5. Write Execution Logic
        # Uses sed to extract the line number matching the array task ID
        f.write(f"cmd=$(sed -n \"${{SLURM_ARRAY_TASK_ID}}p\" {abs_task_file})\n")
        f.write("echo \"Running task: $cmd\"\n")
        f.write("eval $cmd\n")

    print(f"Generated {output_script} with {num_tasks} array tasks.")