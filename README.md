<h1 align="center">
  <p>MARS</p>
  <p>A framework for modelling register-based social networks</p>
</h1>

<p align="center">
  A generative model for multiplex, spatially embedded, affiliation-based networks.
</p>

---

## Overview

MSEAN is a framework for generating multiplex social networks from spatially embedded affiliation structures.

Nodes and affiliations are embedded in a metric space, and node–affiliation relationships are determined by spatial proximity. Network layers represent different types of affiliations, while edges between nodes emerge from shared affiliations.

Experiments are configured through YAML files, making it possible to vary model parameters and generate networks reproducibly.

This is a work in progress, currently available as a [preprint on ArXiv](https://arxiv.org/abs/2608.10946).

## Quick Start

### 1. Prerequisites

Make sure [Git](https://git-scm.com/install/) and [Python](https://www.python.org/downloads/) are installed:

```bash
git --version
python --version
```

### 2. Clone the repository

```bash
git clone https://github.com/IrinaMonaEpure/msean.git
cd msean
```

### 3. Create a virtual environment

**Linux/macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt)**

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### 4. Install MSEAN

Install the package and its dependencies in editable mode:

```bash
pip install -e .
```

### 5. Configure an experiment

Create a YAML configuration file in:

```text
configs/
```

Use [`configs/default.yaml`](configs/default.yaml) as a starting point for the available configuration options.

### 6. Generate networks

Run an experiment with:

```bash
python scripts/generate_network.py
```

Generated networks and experiment outputs are stored in:

```text
runs/
```

## Repository Structure

```text
msean/
├── configs/        # Experiment configuration files
├── runs/           # Generated networks and experiment outputs
├── scripts/        # Scripts for running experiments
├── src/
│   └── msean/      # MSEAN source code
└── pyproject.toml  # Package configuration and dependencies
```

## Model

MSEAN combines three structural components:

* **Multiplexity** — networks consist of multiple affiliation layers.
* **Spatial embedding** — nodes and affiliations are positioned in a metric space, allowing spatial proximity to influence affiliation formation.
* **Affiliation-based structure** — social ties emerge through shared affiliations within each layer.

The spatial freedom parameter controls the influence of spatial proximity on affiliation formation, allowing the model to range from strongly spatially constrained networks to increasingly space-independent networks.

## Configuration

Model parameters are specified in YAML configuration files. A configuration can define properties such as the network size, affiliation layers, spatial embedding, and parameters controlling network generation.

See [`configs/default.yaml`](configs/default.yaml) for an example configuration.

## Output

Each experiment produces output under `runs/`. Depending on the experiment configuration, this can include generated network data and network-property results for subsequent analysis and visualization.
