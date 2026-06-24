# Follow-up Revision Validation And Git Update

## Validation Result

- `python -m compileall src`: passed.
- `python src/06_make_figures.py`: regenerated `fig01_framework_timeline.png` as a workflow figure.
- `python src/11_run_paper_revision_support.py`: refreshed paper-support tables.
- `python src/10_prepare_acml_paper_assets.py`: regenerated paper source, tables, figures, and source zip.
- LaTeX build with `pdflatex -> bibtex -> pdflatex -> pdflatex`: passed.
- Final PDF: 16 pages.
- LaTeX log: no errors, no undefined citations/references, no overfull boxes.
- Remaining warnings: underfull line-break warnings and MiKTeX update advisory only.

## Claim Checks

- Removed the overclaim that history-free weather-soil improves over identity/time baselines.
- Reframed history-free weather-soil as measurable but secondary monitoring signal.
- Preserved headline May-Oct numbers:
  - History-free weather-soil: RMSE 0.889, R2 0.507.
  - Operational: RMSE 0.658, R2 0.730.
- Table 7 now uses `Risk score`.
- Uncertainty text states nominal 80% conformal coverage and validation-period calibration only.
- Stress validation row for held-out years is explicitly an all-window mean.
- `model_ready_panel_improved.csv` remains 4,830 rows and 966 unique region-crop-year yield observations.

## Git Update

- Target remote: `origin/main`.
- Commit hash: recorded in git history after this validation file is committed.
