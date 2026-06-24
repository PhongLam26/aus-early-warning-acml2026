# Follow-up Revision Implementation Log

## Implemented Source Changes

- Updated `src/10_prepare_acml_paper_assets.py` so regenerated paper text no longer overclaims the history-free weather-soil model.
- Updated table generation labels:
  - Table 7 uses `Risk score`.
  - Stress validation uses `Held-out test years, all-window mean`.
  - Uncertainty table includes nominal conformal coverage.
- Updated `src/06_make_figures.py` so `fig01_framework_timeline.png` becomes a workflow figure rather than only a month timeline.

## Interpretation Changes

- History-free weather-soil is now described as positive but secondary monitoring evidence.
- Operational forecasts are described as the preferred deployment configuration for known crop-region histories.
- Soil is described as regional background conditioning, not a causal performance driver.
- Risk classification is framed as ranked analyst review, not an automatic trigger or fully calibrated event probability.

## Pending Validation

- Regenerate figures and paper assets.
- Compile LaTeX.
- Check page count, citations, references, overfull boxes, and key numerical claims.
- Commit and push to `origin/main`.
