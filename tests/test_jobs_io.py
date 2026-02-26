"""Tests for job launcher generation helpers."""

import pytest

from molzen.io.jobs import generate_parallel_launcher


def test_generate_parallel_launcher_default_single_gpu(tmp_path):
    """Default behavior should still map one GPU per job slot."""
    script_path = tmp_path / "run_jobs.sh"

    generate_parallel_launcher(
        cuda_indices=[0, 1],
        jobs_per_gpu=2,
        job_file="tasks.txt",
        output_script=str(script_path),
    )

    content = script_path.read_text()
    assert "Jobs per GPU: 2" in content
    assert "GPUs per job: 1" in content
    assert "Total Concurrency: 4" in content
    assert 'gpu_groups=("0" "1" "0" "1")' in content


def test_generate_parallel_launcher_multi_gpu_jobs(tmp_path):
    """Launcher should assign multiple GPUs per slot when requested."""
    script_path = tmp_path / "run_jobs.sh"

    generate_parallel_launcher(
        cuda_indices=[0, 1, 2, 3, 4, 5, 6, 7],
        jobs_per_gpu=1,
        gpus_per_job=2,
        job_file="tasks.txt",
        output_script=str(script_path),
    )

    content = script_path.read_text()
    assert "Jobs per GPU: 1" in content
    assert "GPUs per job: 2" in content
    assert "Total Concurrency: 4" in content
    assert 'gpu_groups=("0,1" "2,3" "4,5" "6,7")' in content


def test_generate_parallel_launcher_rejects_oversized_gpu_request(tmp_path):
    """Requesting more GPUs per job than available slots should fail."""
    script_path = tmp_path / "run_jobs.sh"

    with pytest.raises(ValueError, match="Not enough total GPU slots"):
        generate_parallel_launcher(
            cuda_indices=[0, 1],
            jobs_per_gpu=1,
            gpus_per_job=3,
            job_file="tasks.txt",
            output_script=str(script_path),
        )
