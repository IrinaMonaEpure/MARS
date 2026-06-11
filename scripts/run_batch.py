from pathlib import Path
import subprocess
import argparse
import sys

parser = argparse.ArgumentParser()

parser.add_argument("--results_dir", type=str, required=True)
parser.add_argument("--logs_dir", type=str, required=True)
parser.add_argument("--n_runs", type=int, required=True)
parser.add_argument("--max_parallel", type=int, required=True)

args = parser.parse_args()

results_dir = Path(args.results_dir)
results_dir.mkdir(parents=True, exist_ok=True)

logs_dir = Path(args.logs_dir)
logs_dir.mkdir(parents=True, exist_ok=True)

for batch_start in range(0, args.n_runs, args.max_parallel):

    batch_end = min(
        batch_start + args.max_parallel,
        args.n_runs,
    )

    print(
        f"Starting seeds {batch_start}-{batch_end - 1}",
        flush=True,
    )

    processes = []

    for seed in range(batch_start, batch_end):

        log_path = logs_dir / f"seed{seed}.log"
        log_file = open(log_path, "w")

        p = subprocess.Popen(
            [
                sys.executable,
                "experiment.py",
                "--seed",
                str(seed),
                "--results_dir",
                str(results_dir),
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

        processes.append((seed, p, log_file))

        print(
            f"Started seed {seed}, PID={p.pid}, log={log_path}",
            flush=True,
        )

    for seed, p, log_file in processes:

        return_code = p.wait()
        log_file.close()

        if return_code == 0:
            print(f"Finished seed {seed}", flush=True)
        else:
            print(
                f"Seed {seed} failed with return code {return_code}",
                flush=True,
            )

print("All experiments finished.", flush=True)
