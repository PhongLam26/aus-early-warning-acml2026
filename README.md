# Australian Winter Crop Early-Warning Project

This repository contains the paper draft, reproducible scripts, processed data panels, and result artifacts for:

**Stage-Aware Early Warning of Australian Winter Crop Yield Shortfall Using Daily Weather and Soil Data**

The current manuscript is prepared in anonymous ACML/JMLR style for ACML 2026 conference-track submission.

## Start Here

- Read the compiled main paper: `outputs/overleaf/acml2026_aus_early_warning_overleaf/AUS_DAP.pdf`
- Read the supplementary material: `outputs/overleaf/acml2026_aus_early_warning_overleaf/AUS_DAP_sup.pdf`
- Upload or rebuild the Overleaf source bundle: `outputs/overleaf/acml2026_aus_early_warning_overleaf/`
- Downloadable Overleaf zip: `outputs/overleaf/acml2026_aus_early_warning_overleaf.zip`
- Build and validation notes: `outputs/reports/ACML_PAPER_VALIDATION_LOG.md`
- Main modeling evidence: `outputs/tables/sota_baseline_comparison.csv`, `outputs/tables/paper_revision_fixed_model_lead_time.csv`, and `outputs/tables/round2_paper_table_validation.csv`

## What Is Included

- `outputs/overleaf/acml2026_aus_early_warning_overleaf/`: current LaTeX manuscript source, bibliography, paper tables, figures, and compiled PDFs.
- `src/`: data preparation, modeling, ablation, paper-support, and paper-generation scripts.
- `config/`: project configuration for crops, regions, forecast windows, splits, and thresholds.
- `data/processed/`: processed weather-stage features, selected soil features, and model-ready panels.
- `data/raw/`: compact raw yield, soil, and per-state daily-weather CSV inputs used to build processed panels.
- `outputs/tables/`: model metrics, ablations, validation checks, classification diagnostics, uncertainty summaries, and paper-support tables.
- `outputs/figures/`: generated figures used in the paper and supplementary result review.
- `outputs/reports/`: method plan, result templates, validation logs, and handoff packages.

The `outputs/safe_versions/` snapshot and raw weather cache are intentionally excluded from git because they are redundant generated artifacts.

## Current Validation Snapshot

- Final main PDF length: 14 pages, within the ACML conference-track page limit.
- LaTeX build: no errors, no undefined citations/references, and no overfull boxes.
- Main panel: 4,830 forecast-window rows from 966 unique region-crop-year yield observations.
- Main May-Oct results in the current manuscript:
  - No-yield-history weather-soil model: RMSE 0.821, R2 0.579.
  - Operational model with yield history: RMSE 0.660, R2 0.728.
- The manuscript limits claims to state-level monitoring for known crop-region histories and does not claim causal attribution, farm-level prediction, insurance-payout suitability, automatic decisions, or unseen-region deployment.

## Reviewer Reproduction Checks

From the repository root:

```powershell
python -m compileall src
python tests/test_no_leakage_baselines.py
python src/12_run_sota_baseline_suite.py
```

The strong-baseline script reads the processed panel in `data/processed/model_ready_panel_improved.csv` and writes controlled internal-baseline metrics to `outputs/tables/sota_baseline_comparison.csv`.

## Rebuild The Current Paper Bundle

From the repository root:

```powershell
cd outputs/overleaf/acml2026_aus_early_warning_overleaf
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error supplementary.tex
pdflatex -interaction=nonstopmode -halt-on-error supplementary.tex
```

The current source bundle already includes `AUS_DAP.pdf` and `AUS_DAP_sup.pdf` for convenience.
