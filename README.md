# Australian Winter Crop Early-Warning Project

This repository contains the paper draft, reproducible scripts, processed data panels, and result artifacts for:

**Stage-Aware Early Warning of Australian Winter Crop Yield Shortfall Using Daily Weather and Soil Data**

The current manuscript is prepared in anonymous ACML/JMLR style for ACML 2026 conference-track submission.

## Start Here

- Read the compiled paper: `paper_acml/main.pdf`
- Submission handoff PDF: `outputs/reports/acml2026_aus_early_warning_anonymous.pdf`
- OpenReview convenience package: `outputs/reports/acml2026_aus_early_warning_openreview_package.zip`
- Source package: `outputs/reports/acml2026_aus_early_warning_anonymous_source.zip`
- Build and validation notes: `outputs/reports/ACML_PAPER_VALIDATION_LOG.md`
- Paper-revision support log: `outputs/paper_revision_support_2026_06_24/01_PAPER_REVISION_SUPPORT_LOG.md`

## What Is Included

- `paper_acml/`: LaTeX manuscript source, bibliography, paper tables, figures, and compiled PDF.
- `src/`: data preparation, modeling, ablation, paper-support, and paper-generation scripts.
- `config/`: project configuration for crops, regions, forecast windows, splits, and thresholds.
- `data/processed/`: processed weather-stage features, selected soil features, and model-ready panels.
- `data/raw/`: compact raw yield, soil, and per-state daily-weather CSV inputs used to build processed panels.
- `outputs/tables/`: model metrics, ablations, validation checks, classification diagnostics, uncertainty summaries, and paper-support tables.
- `outputs/figures/`: generated figures used in the paper and supplementary result review.
- `outputs/reports/`: method plan, result templates, validation logs, and handoff packages.

The `outputs/safe_versions/` snapshot and raw weather cache are intentionally excluded from git because they are redundant generated artifacts.

## Current Validation Snapshot

- Final PDF length: 16 pages, within the ACML prompt limit.
- LaTeX build: no errors, no undefined citations/references, and no overfull boxes.
- Main panel: 4,830 forecast-window rows from 966 unique region-crop-year yield observations.
- Main May-Oct results are unchanged from the frozen Round 2 evidence:
  - History-free weather-soil model: RMSE 0.889, R2 0.507.
  - Operational model with lag-yield history: RMSE 0.658, R2 0.730.
- The manuscript explicitly limits claims to state-level monitoring and does not claim causal attribution, farm-level prediction, insurance-payout suitability, or automatic policy decisions.

## Rebuild Commands

From the repository root:

```powershell
python -m compileall src
python src/11_run_paper_revision_support.py
python src/10_prepare_acml_paper_assets.py
cd paper_acml
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The paper generator refreshes `paper_acml/main.tex`, `paper_acml/acml26.bib`, paper tables/figures, and the anonymous source zip in `outputs/reports/`.
