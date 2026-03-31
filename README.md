<h1 align="center">
  MSEAN
</h1>

</br>

<h2> Quick Start </h2>

Create and activate virtual environment.
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

Install source code and dependencies.
```bash
pip install -e .
```

Store your configuration file in ```configs/```, following the structure of ```configs/default.yaml```. Run your experiment:
```bash
python scripts/generate_network.py
```

You will find your results in ```runs/```.