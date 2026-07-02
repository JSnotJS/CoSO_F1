import json
import re
import subprocess
import sys
import time
from pathlib import Path


RUNS_PER_BATCH = 25
SEED_BASE = 100_000

ROOT_DIR = Path(__file__).resolve().parents[1]
COSO_SCRIPT = ROOT_DIR / "CoSO.py"
RESULTS_DIR = ROOT_DIR / "results"


COMMON_ARGS = [
    "--verbose", "0",
    "--max_evals", "30000",
    "--elite_pop_size", "10",
    "--genotypes_file", "./population/f1_dataset.tsv",
]


# Each entry defines one batch. A batch launches RUNS_PER_BATCH CoSO processes in parallel.
# Add/edit entries here when preparing experiment series.
BATCH_CONFIGS = [
    {
        "name": "spatial_0_direction_0",
        "args": [
            *COMMON_ARGS,
            "--thresh_delta", "0",
            "--thresh_cog", "0",
        ],
    },
    {
        "name": "spatial_0_direction_30",
        "args": [
            *COMMON_ARGS,
            "--thresh_delta", "0",
            "--thresh_cog", ".5",
        ],
    },
    {
        "name": "spatial_0_direction_60",
        "args": [
            *COMMON_ARGS,
            "--thresh_delta", "0",
            "--thresh_cog", "1",
        ],
    },
    {
        "name": "spatial_0_direction_90",
        "args": [
            *COMMON_ARGS,
            "--thresh_delta", "0",
            "--thresh_cog", "1.5",
        ],
    },
    {
        "name": "spatial_1_direction_0",
        "args": [
            *COMMON_ARGS,
            "--thresh_delta", "1",
            "--thresh_cog", "0",
        ],
    },
    {
        "name": "spatial_1_direction_30",
        "args": [
            *COMMON_ARGS,
            "--thresh_delta", "1",
            "--thresh_cog", ".5",
        ],
    },
    {
        "name": "spatial_1_direction_60",
        "args": [
            *COMMON_ARGS,
            "--thresh_delta", "1",
            "--thresh_cog", "1",
        ],
    },
    {
        "name": "spatial_1_direction_90",
        "args": [
            *COMMON_ARGS,
            "--thresh_delta", "1",
            "--thresh_cog", "1.5",
        ],
    },
    {
        "name": "spatial_2_direction_0",
        "args": [
            *COMMON_ARGS,
            "--thresh_delta", "2",
            "--thresh_cog", "0",
        ],
    },
    {
        "name": "spatial_2_direction_30",
        "args": [
            *COMMON_ARGS,
            "--thresh_delta", "2",
            "--thresh_cog", ".5",
        ],
    },
    {
        "name": "spatial_2_direction_60",
        "args": [
            *COMMON_ARGS,
            "--thresh_delta", "2",
            "--thresh_cog", "1",
        ],
    },
    {
        "name": "spatial_2_direction_90",
        "args": [
            *COMMON_ARGS,
            "--thresh_delta", "2",
            "--thresh_cog", "1.5",
        ],
    },
    {
        "name": "spatial_3_direction_0",
        "args": [
            *COMMON_ARGS,
            "--thresh_delta", "3",
            "--thresh_cog", "0",
        ],
    },
    {
        "name": "spatial_3_direction_30",
        "args": [
            *COMMON_ARGS,
            "--thresh_delta", "3",
            "--thresh_cog", ".5",
        ],
    },
    {
        "name": "spatial_3_direction_60",
        "args": [
            *COMMON_ARGS,
            "--thresh_delta", "3",
            "--thresh_cog", "1",
        ],
    },
    {
        "name": "spatial_3_direction_90",
        "args": [
            *COMMON_ARGS,
            "--thresh_delta", "3",
            "--thresh_cog", "1.5",
        ],
    },
]


def safe_name(value):
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_.-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_") or "batch"


def build_command(batch_args, batch_dir, batch_index, run_index):
    seed = SEED_BASE + batch_index * 10_000 + run_index
    result_prefix = batch_dir / f"run_{run_index:02d}_"

    return [
        sys.executable,
        str(COSO_SCRIPT),
        *batch_args,
        "--seed", str(seed),
        "--filename", str(result_prefix),
    ]


def write_manifest(batch_dir, batch_name, batch_args, batch_index):
    manifest = {
        "batch_index": batch_index,
        "batch_name": batch_name,
        "runs_per_batch": RUNS_PER_BATCH,
        "seed_base": SEED_BASE + batch_index * 10_000,
        "coso_script": str(COSO_SCRIPT),
        "args": batch_args,
        "example_command": " ".join(build_command(batch_args, batch_dir, batch_index, 0)),
    }

    with (batch_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def launch_run(batch_args, batch_dir, batch_index, run_index):
    logs_dir = batch_dir / "process_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    stdout_path = logs_dir / f"run_{run_index:02d}.stdout.txt"
    stderr_path = logs_dir / f"run_{run_index:02d}.stderr.txt"

    stdout = stdout_path.open("w", encoding="utf-8")
    stderr = stderr_path.open("w", encoding="utf-8")

    process = subprocess.Popen(
        build_command(batch_args, batch_dir, batch_index, run_index),
        cwd=ROOT_DIR,
        stdout=stdout,
        stderr=stderr,
        text=True,
    )

    return {
        "run_index": run_index,
        "process": process,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
    }


def close_streams(run):
    run["stdout"].close()
    run["stderr"].close()


def terminate_runs(runs):
    for run in runs:
        if run["process"].poll() is None:
            run["process"].terminate()
    for run in runs:
        run["process"].wait()
        close_streams(run)


def run_batch(batch_index, config):
    batch_name = safe_name(config["name"])
    batch_args = list(config["args"])
    batch_dir = RESULTS_DIR / f"{batch_index:02d}_{batch_name}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(batch_dir, batch_name, batch_args, batch_index)

    print(f"Starting batch {batch_index:02d}: {batch_name}")
    print(f"Results: {batch_dir}")
    print("Example command:", " ".join(build_command(batch_args, batch_dir, batch_index, 0)))

    started_at = time.perf_counter()
    runs = [launch_run(batch_args, batch_dir, batch_index, i) for i in range(RUNS_PER_BATCH)]

    failures = []
    try:
        for run in runs:
            return_code = run["process"].wait()
            close_streams(run)
            if return_code != 0:
                failures.append((run["run_index"], return_code, run["stderr_path"]))
            print(f"  run {run['run_index']:02d} finished with code {return_code}")
    except KeyboardInterrupt:
        print("Interrupted. Terminating running CoSO processes...")
        terminate_runs(runs)
        raise

    elapsed_s = time.perf_counter() - started_at
    print(f"Finished batch {batch_name} in {elapsed_s:.2f}s")

    if failures:
        print(f"Batch {batch_name} failures:")
        for run_index, return_code, stderr_path in failures:
            print(f"  run {run_index:02d}: code {return_code}, stderr: {stderr_path}")
        return False

    return True


def main():
    print(f"Configured batches: {len(BATCH_CONFIGS)}")
    print(f"Parallel CoSO instances per batch: {RUNS_PER_BATCH}")
    print(f"All batch folders will be created under: {RESULTS_DIR}")

    all_ok = True
    for batch_index, config in enumerate(BATCH_CONFIGS):
        all_ok = run_batch(batch_index, config) and all_ok

    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
