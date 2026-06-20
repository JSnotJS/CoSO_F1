import subprocess
import sys
import time
from pathlib import Path


RUNS = 30

BASE_ARGS = [
    "--verbose", "0",
    "--max_evals", "25000",
    "--elite_pop_size", "10",
    "--thresh_delta", "2",
    "--thresh_cog", "60",
    "--genotypes_file", "./population/f1_dataset.tsv",
]

ROOT_DIR = Path(__file__).resolve().parents[1]
COSO_SCRIPT = ROOT_DIR / "CoSO.py"
LAUNCH_LOG_DIR = ROOT_DIR / "launch_tests_logs"


def build_command():
    return [sys.executable, str(COSO_SCRIPT), *BASE_ARGS]


def launch_run(index):
    LAUNCH_LOG_DIR.mkdir(parents=True, exist_ok=True)

    stdout_path = LAUNCH_LOG_DIR / f"run_{index:02d}.stdout.txt"
    stderr_path = LAUNCH_LOG_DIR / f"run_{index:02d}.stderr.txt"

    stdout = stdout_path.open("w", encoding="utf-8")
    stderr = stderr_path.open("w", encoding="utf-8")

    process = subprocess.Popen(
        build_command(),
        cwd=ROOT_DIR,
        stdout=stdout,
        stderr=stderr,
        text=True,
    )

    return {
        "index": index,
        "process": process,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
    }


def close_streams(run):
    run["stdout"].close()
    run["stderr"].close()


def main():
    print(f"Launching {RUNS} parallel CoSO runs")
    print("Command:", " ".join(build_command()))
    print(f"Process logs: {LAUNCH_LOG_DIR}")

    started_at = time.perf_counter()
    runs = [launch_run(i) for i in range(RUNS)]

    failures = []
    try:
        for run in runs:
            return_code = run["process"].wait()
            close_streams(run)

            if return_code != 0:
                failures.append((run["index"], return_code, run["stderr_path"]))

            print(f"Run {run['index']:02d} finished with code {return_code}")
    except KeyboardInterrupt:
        print("Interrupted. Terminating child processes...")
        for run in runs:
            if run["process"].poll() is None:
                run["process"].terminate()
        for run in runs:
            run["process"].wait()
            close_streams(run)
        raise

    elapsed_s = time.perf_counter() - started_at
    print(f"Finished {RUNS} runs in {elapsed_s:.2f}s")
    print(f"Config: {'_'.join(BASE_ARGS[1::2])}") # tylko wartości parametrów

    if failures:
        print("Failures:")
        for index, return_code, stderr_path in failures:
            print(f"  run {index:02d}: code {return_code}, stderr: {stderr_path}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
