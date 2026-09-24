<h1 align="center">
  <p>MARS</p>
  <p>A framework for modelling register-based social networks</p>
</h1>

This repository contains the implementation of the Multiplex Affiliation-based Random Spatially-embedded (MARS) graph framework, which replicates the construction method of register-based social networks. This codebase currently allows for the reproduction of results included in:
> Hamilton, K., Epure, I., & Takes, F. (2026). MARS: A framework for modelling register-based social networks. arXiv preprint arXiv:2608.10946.

This is currently accessible as a [preprint](https://arxiv.org/abs/2608.10946). A user guide for the framework is under work.

## Quick Start

### 1. Environment Setup
This project requires prior setup of [Git](https://git-scm.com/), [Conda](https://anaconda.org/channels/anaconda/packages/conda/overview), and [Python](https://www.python.org/) 3.12 or higher. You can clone this repository and set up a Conda environment with the dependencies specified in `pyproject.toml` in the following way:
```bash
git clone https://github.com/IrinaMonaEpure/MARS.git
cd MARS

conda create -n mars python=3.12
conda activate mars
python -m pip install -e .
```

### 2. Running MARS
The experiments detailed in our preprint can be run using the files inside the `scripts` folder. They are configured based on the file `configs/final_draft.yaml` and by setting variables `cfg`, `std_vals`, and `alpha_vals` inside file `scripts/experiment.py`.
- `cfg` specifies which configuration file is used: `cfg = load_config(root / "configs" / "final_draft.yaml")`;
- `std_vals` is the list of standard deviation (sigma) values used: `std_vals = [0.1, 0.125, 0.15, 0.175, 0.2]`;
- `alpha_vals` is the list of spatial freedom (alpha) values used: `alpha_vals = [(1/2)**(i/2) for i in range(0, 21)]`.
```bash
cd scripts

# run_batch.py calls experiment.py n_runs times, launching max_parallel jobs in parallel at once
python run_batch.py --results_dir ../outputs/batch_run/results --logs_dir ../outputs/batch_run/logs --n_runs 100 --max_parallel 25

# stats_generic.py creates .csv files detailing the average distributions of a selected property in the experiment result files
python stats_generic.py --results_dir ../outputs/batch_run/results --output ../outputs/stats/degree_stats.csv --property DEGREE_DISTRIBUTION

# aggregate_results.py aggregates all experiment result files into a single .pkl file
python aggregate_results.py --results_dir ../outputs/batch_run/results --output ../outputs/batch_run/aggregate_results/batch_experiment_results.pkl
```

### 3. Generating Plots and Tables
The figures and tables included in our preprint can be generated using the Jupyter notebooks inside folder `notebooks`. If you did not run the full experiments as instructed in the previous step, you can still run the notebooks using the provided input files found inside folder `inputs`. However, if you did run the commands above, you can make the following changes to use your locally generated data:
- inside `notebooks/fit_alpha_and_std_on_degree_dist.ipynb`, set `summary_path` to `../outputs/batch_run/stats/degree_stats.csv`;
- inside `notebooks/final_draft.ipynb`, set `results_path` to  `..outputs/batch_run/aggregate_results/batch_experiment_results.pkl`.

The outputs of the notebooks are saved under `outputs/plots` and `outputs/tex`.
