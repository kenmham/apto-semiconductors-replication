# Semiconductor capability reproduction

Code and data for the figures, empirical tables, and reported empirical analyses in *Punctuated Equilibrium in Technological Capabilities: Commercial Selection and the Predictability of Frontier Advances*.

## Run

Use Python 3.14.5. From this directory:

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python reproduce.py
```

The full run includes the timing and stratification permutation tests and checks the generated outputs against the manuscript’s reported values. Outputs are written under `outputs/`.

## Outputs

- `outputs/figures/figure_01.png` through `figure_11.png`: Figures 1–11. The trajectory figures also have PDF versions.
- `outputs/data/concentration.json`: the episode and gain-concentration statistics in Table 1.
- `outputs/data/lead_ratios.csv`: the retained capability lead ratios in Table B.1.
- `outputs/data/retained_summary.json` and `outputs/diagnostics/`: forecast, calibration, annual-fold, and omission comparisons in Section 4.4.
- `outputs/data/sample_sensitivities.json`: comparisons of the original 206-capability panel, retained 204-capability panel, and the additional exclusion of the flagged power-management current series.
- `outputs/data/timing_stratification/`: timing statistics, permutation comparisons, scored predictions, shoulder-depth comparisons, and product strata in Section 4.5.
- `outputs/data/coupling.json` and `coupling_changes.csv`: the two coupling cases and pooled frontier-stall comparisons in Section 4.6.
- `outputs/data/pilots.json`, `pilot_policy_vs_truth.csv`, and `pilot_attribution_scores.csv`: the saved-response pilot analyses in Appendix G.

Tables 2 and C.1 describe measurements and methods.

To rerun an individual analysis after the full run:

```sh
.venv/bin/python reproduce.py --stage sensitivity
.venv/bin/python reproduce.py --stage timing-stratification
.venv/bin/python reproduce.py --stage coupling
.venv/bin/python reproduce.py --stage pilots
```

## Data

`data/current.parquet` contains the analysis catalog. `data/feature_selection.csv` fixes the selected capabilities and their directions. `data/shoulder_fits.csv` contains the capability-level bootstrap estimates used by the default run.

To recompute the shoulder bootstrap and fitted null estimates separately:

```sh
.venv/bin/python reproduce.py --stage shoulders
```

## Historical language-model pilots

`data/pilots/` contains the historical pilot panel, forecast responses for both prompt versions, ablation responses, attribution responses, and batch comparisons. These retain the original pilot sample. The main forecast analysis uses the retained 204-capability panel.

The pilot stage scores saved responses without model or API calls.

`data/pilots/policy_vs_truth.csv` preserves the rounded historical feature–outcome association table used for the reported rank correlation. The stage also recomputes these associations from the original 206-capability panel and saves them separately as `outputs/data/pilot_recomputed_associations.csv`. Both calculations give the reported rank correlation, approximately −0.07; some individual recomputed coefficients differ from the historical table.

`analysis/pilot_prompts.py` contains the original prompt templates and snapshot construction. `data/pilots/settings.json` records the requested model and available settings.

The saved forecast raw text can be truncated. The ablation and attribution files retain parsed responses rather than complete raw output. The resolved model revision and model sampling settings were not recorded, so these files support reproduction of the reported scores rather than an exact rerun of the original model calls.
