# Codex Fix Checklist Implementation Log

Source checklist: `Codex_fix_checklist_bai_Uc_Early_Warning.docx`.

## Implemented Fixes

- Replaced remaining visible manuscript use of the old weather-anomaly feature wording with `train-derived weather-deviation features`.
- Softened the introduction claim about evidence strength to measurable evidence of within-season monitoring signal.
- Replaced the lead-time fixed-model cross-reference with `Appendix A`, avoiding the wrong rendered appendix-number issue.
- Added a clear explanation that `Dev` means train-derived weather-deviation features.
- Regenerated Figure 5 feature-group labels so the visible label is `weather_dev`, not `weather_anomaly`.
- Added BibTeX accent normalization in the paper generator:
  - `Fran{\c c}ois`
  - `Herv{\'e}`
  - `M{\"u}ller`
- Refreshed `paper_acml/main.pdf`, the anonymous handoff PDF, the source zip, and the OpenReview convenience package.

## Validation

- `python -m compileall src`: passed.
- `python src/11_run_paper_revision_support.py`: passed.
- `python src/10_prepare_acml_paper_assets.py`: passed.
- LaTeX build with `pdflatex -> bibtex -> pdflatex -> pdflatex`: passed.
- Final PDF: 16 pages.
- LaTeX log: no errors, no undefined citations/references, no overfull boxes.
- PDF text check confirms:
  - `Appendix A verifies` appears.
  - `Dev denotes train-derived weather-deviation features` appears.
  - `train-derived weather-deviation features` appears.
  - The old appendix-number rendering, old weather-anomaly feature wording, and over-strong evidence wording do not appear.
