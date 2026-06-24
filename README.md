# AML allo-HCT Risk Calculator

Minimal Streamlit deployment package for the lightweight AML allo-HCT online risk calculator.

This deployment version uses a small JSON weight file instead of a large local model payload. The full manuscript/local analysis workflow can remain separate, while this online calculator is optimized for free Streamlit Community Cloud deployment.

## Files

- `app.py`: Streamlit application and JSON-based inference logic.
- `models/risk_calculator_weights.json`: compact model weights, preprocessing metadata, baseline estimates, and risk-strata thresholds.
- `requirements.txt`: minimal Python dependencies.
- `.streamlit/config.toml`: app theme.

## Why This Version Is Lightweight

- No `.joblib` model file.
- No Git LFS.
- No GitHub Release asset.
- No `MODEL_URL` secret.
- No `scikit-survival`, `scikit-learn`, or `joblib` dependency.
- Online prediction uses compact JSON weights bundled with the app.

Risk strata are based on outcome-specific model-score tertiles in the development cohort.

## Deploy With GitHub + Streamlit Community Cloud

1. Copy only the contents of this folder into the root of your GitHub repository, for example `aml-allohct-risk-calculator`.
2. Confirm the repository contains these files:

```text
app.py
requirements.txt
README.md
.gitignore
.streamlit/config.toml
models/risk_calculator_weights.json
```

3. Confirm the repository does not contain any `.joblib` model file:

```bash
git status
git ls-files | grep joblib
```

4. Commit and push:

```bash
git add app.py requirements.txt README.md .gitignore .streamlit/config.toml models/risk_calculator_weights.json
git commit -m "Deploy lightweight AML allo-HCT risk calculator"
git push
```

5. Open [Streamlit Community Cloud](https://share.streamlit.io/), choose **Create app** or **Deploy an app**, and select the GitHub repository.
6. Set the main file path to:

```text
app.py
```

7. Choose Python 3.11 if Streamlit offers a Python-version option.
8. Deploy the app. No secrets are required.

## Local Smoke Test

From this folder:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app should load the JSON weights, show the 10 predictor inputs, and calculate four outcome-specific 12-, 24-, and 36-month risks.

## Research-Only Disclaimer

This calculator is a research prototype. It should not be used for bedside decision-making without prospective validation, local recalibration, and final clinical category review.
