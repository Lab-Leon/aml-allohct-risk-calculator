# AML allo-HCT Risk Calculator — minimal GitHub package

This repository is a lightweight Streamlit distribution of the AML allogeneic haematopoietic cell transplantation (allo-HCT) research calculator. It contains the compact fixed-alpha Cox proportional-hazards (CoxPH) benchmark weights and no patient-level data.

## Included functionality

- CoxPH benchmark predictions at 12, 24, and 36 months for overall survival, relapse, disease-free survival, and TRM/NRM.
- Outcome-specific low/intermediate/high model-score strata.
- The 10 locked pre-transplant or transplant-time inputs used in the manuscript.
- Clinically named category menus aligned with the known-category SHAP displays.
- No more than five choices per categorical predictor.
- No raw source-code labels or source-coded missingness choices in the web interface.

## Input-category convention

Each categorical menu is restricted to clinically named, known categories shown in the streamlined SHAP displays. Source-coded missingness and other unlabelled source levels remain part of the underlying research data architecture but are not presented as patient-input choices in this lightweight web interface.

## Deliberate lightweight boundary

The optional 1000-tree RSF research models are **not** included because their serialized artifacts are several gigabytes and exceed normal GitHub/Streamlit Cloud repository limits. The interface therefore shows the CoxPH benchmark only.

To restore the RSF option in a controlled local environment, use the full internal calculator distribution together with the original `models/rsf1000/` artifacts and its corresponding dependencies. Do not add those artifacts to this repository without an appropriate storage and access plan.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

For Streamlit Community Cloud, select `app.py` as the entry point.

## Research-use statement

This calculator is intended for research communication, cohort stratification, and recalibration studies. Prospective evaluation and setting-specific recalibration are important before patient-level implementation. The relapse and TRM/NRM outputs are cause-specific model estimates, not native cumulative-incidence-function probabilities; all endpoint outputs are interpreted separately and must not be combined to reconstruct disease-free survival.

## Package contents

```text
app.py                               Streamlit application
models/risk_calculator_weights.json  Compact CoxPH coefficients, baselines, thresholds and input metadata
.streamlit/config.toml               Visual theme configuration
requirements.txt                     Minimal runtime dependencies
```
