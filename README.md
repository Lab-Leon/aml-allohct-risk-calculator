# AML allo-HCT Research Model Explorer

This repository contains the reviewer-facing Streamlit implementation accompanying the AML allo-HCT prognostic-model study. It provides research-only exploration of the development-selected L2-penalized Cox, LASSO-Cox, and random survival forest models for overall survival, relapse, disease-free survival, and transplant-related/non-relapse mortality.

## Interpretation boundary

- Outputs are for research communication, cohort stratification, and independent validation or recalibration studies.
- The application is not validated or authorized for bedside treatment decisions and does not recommend treatment.
- OS and DFS survival probabilities are uncalibrated research estimates; the manuscript reports systematic external miscalibration.
- Relapse and TRM/NRM outputs are cause-specific ranking scores, not absolute probabilities or cumulative-incidence estimates.
- The four endpoints were fitted independently and must not be added or combined to reconstruct DFS.
- Individual-prediction confidence intervals are not available.

No patient-level data are included in this repository. The random survival forest files are compact inference artifacts derived from the locked fitted models; the multi-gigabyte research serialization files are not included.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

For Streamlit Community Cloud, select `app.py` as the entry point and Python 3.11 as the runtime.
