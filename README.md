# Semiconductor capability reproduction

Code and data for the figures and empirical tables in *Punctuated Equilibrium in Technological Capabilities: Commercial Selection and the Predictability of Frontier Advances*.

## Run

Use Python 3.14.5. From this directory:

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python reproduce.py
```

Figures 1–11 are written to `outputs/figures/`. Numerical tables, estimates, and intermediate data are written to `outputs/data/`; forecast diagnostics are in `outputs/diagnostics/`.

## Data

`data/current.parquet` contains the analysis catalog. `data/feature_selection.csv` fixes the selected capabilities and their directions. `data/shoulder_fits.csv` contains the capability-level bootstrap estimates used by the default run.

To recompute the shoulder bootstrap and fitted null estimates separately:

```sh
.venv/bin/python reproduce.py --stage shoulders
```
