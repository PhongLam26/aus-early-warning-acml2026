# ACML Paper Validation Log

## Build Status

- Generated manuscript folder: `AUS_project/paper_acml/`.
- Generated anonymous PDF: `AUS_project/paper_acml/main.pdf`.
- Layout-fixed compile artifact: `AUS_project/paper_acml/main_layout_fixed.pdf`.
- Copied PDF for handoff: `AUS_project/outputs/reports/acml2026_aus_early_warning_anonymous.pdf`.
- Generated source zip: `AUS_project/outputs/reports/acml2026_aus_early_warning_anonymous_source.zip`.
- Generated upload convenience package: `AUS_project/outputs/reports/acml2026_aus_early_warning_openreview_package.zip`.

## Compile Command

Run from `AUS_project/paper_acml/`:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

## Final Compile Result

- `main.pdf` compiled successfully.
- Final PDF length: 16 pages.
- ACML limit in prompt: 16 pages including references and appendix.
- Final log has no unresolved citations, no unresolved references, no LaTeX errors, and no overfull boxes.
- Remaining warnings are underfull line-break warnings from figure/table placement and long bibliography URLs. The terminal also reports MiKTeX's local update advisory, which is not a manuscript error.
- `acml26.bib` contains 30 BibTeX entries regenerated from `Tong_hop_ref_bai_Uc_Early_Warning.docx`.
- Hyperlinks are configured with hidden/black link styling, so bibliography URLs do not render in magenta.
- The stress-validation table was compacted to one best model per validation protocol and feature regime, and float barriers were added so the stress heatmap no longer occupies a mostly empty page.
- The P0+P1 revision adds naive baselines, fixed-model lead-time checks, target-threshold sensitivity, a held-out confusion matrix, a top-10 analyst watch list, compact uncertainty diagnostics, and a split feature-importance figure.
- Detailed classification, residual-check, precision-recall, threshold-tradeoff, and interval-coverage artifacts are retained in project outputs; the manuscript appendix keeps only compact tables to stay at the 16-page limit.
- During the latest validation run, `paper_acml/main.pdf`, the handoff PDF, source zip, and OpenReview package were all refreshed successfully from the follow-up revision build.
- Follow-up revision from `Loi_can_sua_tiep_bai_Uc_Early_Warning.docx` was applied: the manuscript no longer claims that the history-free weather-soil model outperforms identity/time or historical baselines.
- Figure 1 was upgraded from a simple timeline to a workflow showing inputs, stage windows, model regimes, and monitoring outputs.

## Anonymous Submission Checks

- `main.tex` uses `\author{}`.
- No author names, emails, affiliations, acknowledgements, or local absolute paths were found in `main.tex` or `acml26.bib`.
- The packaged `jmlr.cls` contains generic class macro strings such as `Email` and `Address`; these are template internals, not manuscript author metadata.

## Scientific Claim Checks

- Main claims use Round 2 frozen results and P0+P1 paper-revision support artifacts generated from existing processed outputs.
- The manuscript separates the history-free weather-soil model from the operational lag-yield model.
- History-free weather-soil is now framed as measurable but secondary monitoring information, not the best forecasting/deployment model.
- Table labels use `Weather+Dev`/`Weather+Dev+Soil` for train-derived weather deviations, reducing anomaly-attribution language.
- Table 7 uses `Risk score` and states that risk scores rank analyst-review watch lists rather than fully calibrated probabilities.
- Uncertainty diagnostics state nominal 80% conformal coverage, validation-period residual calibration, and no test retuning.
- Stress validation labels the held-out test-year row as an all-window mean, avoiding comparison with the May-Oct-only result.
- `model_ready_panel_improved.csv` remains 4,830 window rows and 966 unique region-crop-year yield observations.
- `production_kt` and `area_000ha` remain retained as metadata in the panel but are excluded from the training feature matrices and manuscript evidence.
- Table 2 keeps the frozen May-Oct results: history-free weather-soil RMSE 0.889/R2 0.507 and operational RMSE 0.658/R2 0.730.
- The manuscript states limitations for state-level scope, no causal attribution, no farm-level prediction, and weak leave-one-region-out transfer.

## Paper Revision Support Artifacts

- `AUS_project/src/11_run_paper_revision_support.py`
- `AUS_project/outputs/tables/paper_revision_naive_baselines.csv`
- `AUS_project/outputs/tables/paper_revision_fixed_model_lead_time.csv`
- `AUS_project/outputs/tables/paper_revision_threshold_sensitivity.csv`
- `AUS_project/outputs/tables/paper_revision_threshold_sensitivity_summary.csv`
- `AUS_project/outputs/tables/paper_revision_confusion_matrix.csv`
- `AUS_project/outputs/tables/paper_revision_watch_list_top10.csv`
- `AUS_project/outputs/tables/paper_revision_uncertainty_summary.csv`
- `AUS_project/outputs/tables/paper_revision_feature_importance_split.csv`
- `AUS_project/outputs/figures/fig19_feature_importance_split.png`
- `AUS_project/outputs/paper_revision_support_2026_06_24/01_PAPER_REVISION_SUPPORT_LOG.md`
- `AUS_project/outputs/paper_revision_followup_2026_06_24/01_LOI_CAN_SUA_TIEP_REVISION_PLAN.md`
- `AUS_project/outputs/paper_revision_followup_2026_06_24/02_IMPLEMENTATION_LOG.md`
- `AUS_project/outputs/paper_revision_followup_2026_06_24/03_VALIDATION_AND_GIT_UPDATE.md`
