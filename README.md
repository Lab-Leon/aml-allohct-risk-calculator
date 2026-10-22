# AML allo-HCT Research Model Explorer

Research implementation accompanying the manuscript, using its locked L2-penalized Cox, LASSO-Cox, and random survival forest models. The model inputs comprise age and nine categorical predictors. The development cohort contains 16,824 patients for OS and 16,683 with available relapse, DFS, and TRM/NRM endpoints. The interface uses an integer age grid of 19–81 years within the observed continuous development-age range.

## Research outputs

- Development-score percentile and fixed low, intermediate, or high stratum, with the manuscript cutoffs and tie rules.
- OS and DFS: native, uncalibrated survival probabilities at 12, 24, and 36 months. External validation documented systematic miscalibration; local recalibration is required for patient-level application.
- Relapse and TRM/NRM: cause-specific ranking scores, not event probabilities or cumulative-incidence estimates.
- Endpoints were fitted independently. Relapse and TRM/NRM scores cannot be combined to reconstruct DFS. Individual-prediction confidence intervals are unavailable.

The interface supports research communication and independent validation. Clinical use requires local recalibration and prospective evaluation. No patient records are included. The sorted score reference distributions are unlinked development reference values.

## Reproducibility

The eight Cox artifacts are the original fitted files. Four compact forest artifacts retain the fitted trees and double-precision split thresholds, risk sums, and survival values at the displayed horizons. The source-model checksums are recorded in `outputs/model_metadata.json` and `calculator/compact_rsf/manifest.json`. Synthetic examples in `outputs/reproducibility_examples.json` are provided for output checks; they are not patient records.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

For Streamlit Community Cloud, use Python 3.11 and `app.py` as the entry point.
