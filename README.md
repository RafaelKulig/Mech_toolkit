# Mech_toolkit

A small collection of mechanical engineering helper utilities for beams, stress analysis, fatigue, gears, and numerical methods.

This repository contains the `mech_toolkit` Python package with several modules and sample data files used for calculations and lookup tables.

## Installation

- Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

- Install the package in editable mode (if you plan to modify it):

```bash
pip install -e .
```

If you don't have packaging files, you can still import the package by adding the project root to `PYTHONPATH` or running scripts from the repository root.

## Package Layout

- `mech_toolkit/` — package directory
  - `Beam.py` — beam-related calculations and helpers
  - `Fatigue.py` — fatigue life and S-N curve tools
  - `Gears.py` — gear geometry and strength calculations
  - `Mohr.py` — Mohr's circle / stress transformation helpers
  - `Numerical_Methods.py` — small numerical utilities used by other modules
  - `Stress.py` — stress analysis helpers
  - `data/` — JSON lookup tables used by the package

Data files in `mech_toolkit/data`:

- `load_factors.json` — typical load factors / multipliers
- `reliability.json` — reliability factors and tables
- `surface_finish.json` — surface finish factors used in fatigue calculations

## Quick Start / Examples

These examples are intentionally generic so they work even if specific function names differ. Replace the example calls with the actual function/class names found in each module.

1) Inspect the package and modules:

```python
import mech_toolkit
import importlib
importlib.reload(mech_toolkit)
print('Available attributes in mech_toolkit:', [name for name in dir(mech_toolkit) if not name.startswith('__')])

# Inspect a module, e.g. Beam
from mech_toolkit import Beam
print('Beam contents:', [name for name in dir(Beam) if not name.startswith('__')])
```

2) Load the JSON lookup data bundled with the package (portable method):

```python
import json
import pkgutil

data_bytes = pkgutil.get_data('mech_toolkit', 'data/load_factors.json')
load_factors = json.loads(data_bytes)
print(load_factors)
```

3) Example pattern to run a calculation (replace with real function names):

```python
from mech_toolkit import Beam

# If Beam exposes a function like `bending_stress`, you'd call it like:
# result = Beam.bending_stress(M=1000, c=0.05, I=1.2e-6)
# print('Bending stress (Pa):', result)

# If Beam exposes classes, instantiate and call methods:
# beam = Beam.BeamSection(width=0.02, height=0.04)
# print(beam.second_moment())
```

## Development

- Use `pip install -e .` to install in editable mode when you add packaging metadata (`setup.py` / `pyproject.toml`).
- Add unit tests under a `tests/` folder and run them with `pytest`.

## Contributing

Contributions, bug reports, and feature requests are welcome. Please open an issue describing the change and follow up with a pull request.

## License

See the `LICENSE` file in the repository root for licensing details.