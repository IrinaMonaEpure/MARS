<h1 align="center">
  <p>MSEAN</p>
  <p>Multiplex Spatially Embedded Affiliation-Based Network</p>
</h1>

<h2> Quick Start </h2>

Open a terminal. Make sure you have [Git](https://git-scm.com/install/) and [Python](https://www.python.org/downloads/) installed:
```bash
git --version
python --version
```

Clone the MSEAN repository:
```bash
git clone https://github.com/IrinaMonaEpure/msean.git
cd msean
```

Create and activate virtual environment:
```bash
# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Windows (cmd)
python -m venv .venv
.venv\Scripts\activate.bat
```

Install source code and dependencies:
```bash
pip install -e .
```

Store your configuration file in ```configs/```, following the structure of ```configs/default.yaml```. Run your experiment:
```bash
python scripts/generate_network.py
```

You will find the output files in ```runs/```.