from pathlib import Path
import subprocess

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

processes = []

for seed in range(10):

    p = subprocess.Popen(
        [
            "python",
            "batch_experiment.py",
            "--seed",
            str(seed),
            "--results_dir",
            str(results_dir),
        ]
    )

    processes.append(p)

print(f"Started {len(processes)} jobs.")