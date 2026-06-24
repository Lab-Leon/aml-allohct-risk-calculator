from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


APP_ROOT = Path(__file__).resolve().parent
WEIGHT_CANDIDATES = [
    APP_ROOT / "models" / "risk_calculator_weights.json",
    APP_ROOT / "risk_calculator_weights.json",
    APP_ROOT / "models" / "coxph_calculator_weights.json",
    APP_ROOT / "coxph_calculator_weights.json",
]

ESTIMATE_LABELS = {
    "OS": "24-month model-estimated death risk",
    "Relapse": "24-month cause-specific relapse estimate",
    "DFS": "24-month model-estimated relapse/death risk",
    "TRM_NRM": "24-month cause-specific TRM/NRM estimate",
}

RISK_STYLE = {
    "Low": {"label": "Low risk", "class": "risk-low", "rank": 0},
    "Intermediate": {"label": "Intermediate risk", "class": "risk-mid", "rank": 1},
    "High": {"label": "High risk", "class": "risk-high", "rank": 2},
}

FEATURE_DISPLAY = {
    "age": {
        "label": "Age at transplantation",
        "help": "Recipient age at allogeneic hematopoietic cell transplantation, in years.",
    },
    "sex": {
        "label": "Recipient sex",
        "help": "Biologic sex recorded for the transplant recipient.",
        "options": {1.0: "Male", 2.0: "Female"},
    },
    "kps": {
        "label": "Karnofsky performance status",
        "help": "Functional status before conditioning or transplantation.",
        "options": {0.0: "KPS <90", 1.0: "KPS >=90"},
    },
    "hct_ci": {
        "label": "HCT-CI comorbidity score",
        "help": "Hematopoietic Cell Transplantation-specific Comorbidity Index or grouped comorbidity score before transplant.",
        "options": {0.0: "HCT-CI 0", 1.0: "HCT-CI 1-2", 2.0: "HCT-CI >=3", 98.0: "Not available"},
    },
    "cytogenetics": {
        "label": "Cytogenetic risk group",
        "help": "Cytogenetic or ELN-style risk grouping available before transplant.",
        "options": {2.0: "Favorable", 3.0: "Intermediate", 4.0: "Poor/adverse", 8.0: "Not tested/unknown"},
    },
    "mrd": {
        "label": "Measurable residual disease before transplant",
        "help": "MRD status assessed before allogeneic transplantation.",
        "options": {0.0: "MRD negative", 1.0: "MRD positive", 98.0: "Not available/unknown"},
    },
    "donor_type": {
        "label": "Donor type",
        "help": "Donor relationship and HLA-match category.",
        "options": {
            1.0: "HLA-identical sibling donor",
            2.0: "Other/haploidentical related donor",
            3.0: "Matched unrelated donor",
            4.0: "Mismatched/other unrelated donor",
            5.0: "Cord blood donor",
        },
    },
    "graft_type": {
        "label": "Stem-cell source",
        "help": "Primary graft source used for transplantation.",
        "options": {
            1.0: "Bone marrow graft",
            2.0: "Peripheral blood stem cells",
            3.0: "Cord blood graft",
        },
    },
    "conditioning_intensity": {
        "label": "Conditioning intensity",
        "help": "Conditioning-intensity grouping before transplant.",
        "options": {1.0: "MAC, TBI-based", 2.0: "MAC, chemotherapy-based or unspecified", 3.0: "RIC/NMA"},
    },
    "cmv": {
        "label": "Recipient CMV serostatus",
        "help": "Recipient CMV serostatus before transplantation, derived from recipient-only or donor-recipient source fields.",
        "options": {0.0: "Recipient CMV negative", 1.0: "Recipient CMV positive"},
    },
}


st.set_page_config(page_title="AML allo-HCT Risk Calculator", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 2.4rem; padding-bottom: 2.5rem; max-width: 1320px; }
    .app-title {
        display: block; overflow: visible; white-space: normal;
        font-size: 1.9rem; line-height: 1.22; font-weight: 760; margin: .15rem 0 .35rem 0;
    }
    .app-subtitle { color: #4b5563; font-size: .98rem; margin-bottom: 1rem; max-width: 980px; }
    .notice {
        border-left: 4px solid #64748b; background: #f8fafc; color: #334155;
        padding: .78rem .95rem; margin: .6rem 0 1.1rem 0; font-size: .9rem;
    }
    .risk-card {
        border: 1px solid #d8dee6; border-radius: 8px; padding: .9rem .95rem;
        min-height: 136px; background: white; box-shadow: 0 1px 2px rgba(15, 23, 42, .04);
    }
    .risk-low { border-top: 5px solid #3b82c4; background: #f5f9fd; }
    .risk-mid { border-top: 5px solid #d4a23a; background: #fffaf0; }
    .risk-high { border-top: 5px solid #c85d5d; background: #fff6f5; }
    .risk-outcome { color: #475569; font-size: .84rem; font-weight: 650; text-transform: uppercase; letter-spacing: .02em; }
    .risk-tier-label { color: #64748b; font-size: .78rem; margin-top: .35rem; }
    .risk-tier { font-size: 1.28rem; font-weight: 760; margin-top: .02rem; }
    .risk-percent { font-size: 1.9rem; font-weight: 780; margin: .25rem 0 .1rem 0; }
    .risk-note { color: #64748b; font-size: .82rem; }
    .section-label { font-weight: 720; color: #111827; margin: .25rem 0 .35rem 0; }
    div[data-testid="stMetric"] { background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 8px; padding: .55rem .7rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading calculator weights...")
def load_weights() -> dict:
    for path in WEIGHT_CANDIDATES:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))

    expected = "\n".join(f"- `{path.relative_to(APP_ROOT)}`" for path in WEIGHT_CANDIDATES[:2])
    st.error(
        "The calculator weight file was not found. Please upload `risk_calculator_weights.json` "
        f"to one of these locations in the GitHub repository:\n\n{expected}"
    )
    st.stop()


def is_missing(value) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def category_token(value, categories: list[str]) -> str:
    if is_missing(value):
        return "Missing/Unknown"

    raw = value.item() if hasattr(value, "item") else value
    candidates = [str(raw)]

    try:
        numeric = float(raw)
        if numeric.is_integer():
            candidates.append(str(int(numeric)))
        candidates.append(str(numeric))
    except Exception:
        pass

    for token in candidates:
        if token in categories:
            return token
    return candidates[0]


def encode_row(row: pd.Series, outcome_spec: dict) -> np.ndarray:
    values: list[float] = []
    age_value = pd.to_numeric(pd.Series([row.get("age")]), errors="coerce").iloc[0]
    if pd.isna(age_value):
        age_value = outcome_spec["numeric"]["age"]["median"]

    age_spec = outcome_spec["numeric"]["age"]
    values.append((float(age_value) - age_spec["mean"]) / age_spec["scale"])

    for feature in outcome_spec["categorical_features"]:
        categories = outcome_spec["categories"][feature]
        token = category_token(row.get(feature), categories)
        values.extend([1.0 if token == category else 0.0 for category in categories])

    return np.asarray(values, dtype=float)


def risk_group(score: float, thresholds: dict[str, float]) -> str:
    if not np.isfinite(score):
        return "Intermediate"
    if score <= thresholds["low_intermediate"]:
        return "Low"
    if score <= thresholds["intermediate_high"]:
        return "Intermediate"
    return "High"


def risk_rank(group: str) -> int:
    return RISK_STYLE.get(group, RISK_STYLE["Intermediate"])["rank"]


def format_percent(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{100 * value:.1f}%"


def predict_outcome(row: pd.Series, weights: dict, outcome: str) -> dict:
    outcome_spec = weights["outcomes"][outcome]
    horizons = [float(item) for item in weights["horizons"]]
    x = encode_row(row, outcome_spec)
    score = float(np.dot(x, np.asarray(outcome_spec["coef"], dtype=float)))
    exp_score = math.exp(score)

    risks = []
    for horizon in horizons:
        s0 = float(outcome_spec["baseline_survival"][str(int(horizon))])
        risks.append(1.0 - s0**exp_score)

    group = risk_group(score, outcome_spec["risk_thresholds"])
    return {
        "Outcome key": outcome,
        "Outcome": weights["outcome_labels"][outcome],
        "Endpoint": weights["outcome_descriptions"][outcome],
        "Risk group": group,
        "Risk tier": RISK_STYLE[group]["label"],
        "12-month risk": risks[0],
        "24-month risk": risks[1],
        "36-month risk": risks[2],
        "Risk score": score,
        "Low/Intermediate threshold": outcome_spec["risk_thresholds"]["low_intermediate"],
        "Intermediate/High threshold": outcome_spec["risk_thresholds"]["intermediate_high"],
    }


def select_category(feature: str, meta: dict):
    display_meta = FEATURE_DISPLAY.get(feature, {})
    clinical_options = display_meta.get("options", {})
    model_values = [item["value"] for item in meta.get("options", [])]
    available_options = [
        {"label": label, "value": value}
        for value, label in clinical_options.items()
        if value in model_values
    ]
    options = [{"label": "Not available / unknown", "value": np.nan}] + available_options
    selected = st.selectbox(
        display_meta.get("label", meta["label"]),
        options=list(range(len(options))),
        format_func=lambda i: options[i]["label"],
        help=display_meta.get("help", meta.get("description")),
    )
    return options[selected]["value"]


def input_feature(feature: str, meta: dict):
    if meta["type"] == "number":
        min_v = float(max(0.0, np.floor(meta.get("min", 0.0))))
        max_v = float(np.ceil(meta.get("max", 100.0)))
        default = float(meta.get("median", (min_v + max_v) / 2))
        display_meta = FEATURE_DISPLAY.get(feature, {})
        return st.number_input(
            display_meta.get("label", meta["label"]),
            min_value=min_v,
            max_value=max_v,
            value=default,
            step=1.0,
            help=display_meta.get("help", meta.get("description")),
        )
    return select_category(feature, meta)


def prediction_table(row: pd.DataFrame, weights: dict) -> pd.DataFrame:
    input_row = row.iloc[0]
    return pd.DataFrame([predict_outcome(input_row, weights, outcome) for outcome in weights["outcomes"]])


def risk_card(row: pd.Series) -> None:
    style = RISK_STYLE[row["Risk group"]]
    st.markdown(
        f"""
        <div class="risk-card {style['class']}">
            <div class="risk-outcome">{row['Outcome']}</div>
            <div class="risk-tier-label">Model-score stratum</div>
            <div class="risk-tier">{style['label']}</div>
            <div class="risk-percent">{format_percent(row['24-month risk'])}</div>
            <div class="risk-note">{ESTIMATE_LABELS[row['Outcome key']]}<br>{row['Endpoint']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def chart_data(results: pd.DataFrame) -> pd.DataFrame:
    long = results.melt(
        id_vars=["Outcome"],
        value_vars=["12-month risk", "24-month risk", "36-month risk"],
        var_name="Horizon",
        value_name="Predicted risk",
    )
    long["Horizon"] = long["Horizon"].str.replace("-month risk", " months", regex=False)
    return long.pivot(index="Horizon", columns="Outcome", values="Predicted risk").loc[["12 months", "24 months", "36 months"]]


weights = load_weights()
features = weights["features"]
feature_meta = weights["feature_metadata"]

st.markdown('<div class="app-title">AML allo-HCT Risk Calculator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Lightweight web calculator for 12-, 24-, and 36-month post-transplant event risk and low/intermediate/high risk stratification after allogeneic HCT for AML.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="notice">
    Research prototype only. This lightweight web version uses compact JSON weights for free Streamlit deployment.
    The full manuscript-level modeling and interpretation workflow remains part of the local research analysis.
    Risk strata are based on outcome-specific model-score tertiles in the development cohort, not fixed probability
    cut-points.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Patient and transplant inputs")
    st.caption("Use values known before or at transplantation. Leave unavailable fields as not available / unknown.")
    values = {}
    for feature in features:
        values[feature] = input_feature(feature, feature_meta[feature])
    st.caption("Categorical predictors use concise clinical labels after source-specific public-code recoding.")
    calculate = st.button("Calculate risk", type="primary", width="stretch")

input_row = pd.DataFrame([{feature: values.get(feature, np.nan) for feature in features}])

if calculate:
    results = prediction_table(input_row, weights)
    highest = results.sort_values(["Risk group"], key=lambda s: s.map(risk_rank), ascending=False).iloc[0]

    st.markdown('<div class="section-label">Risk stratification summary</div>', unsafe_allow_html=True)
    st.caption(
        "Each stratum is outcome-specific and based on the model score. A 24-month estimate is an absolute "
        "model estimate, whereas low/intermediate/high describes the patient's relative score within that outcome."
    )
    cols = st.columns(4)
    for col, (_, row) in zip(cols, results.iterrows()):
        with col:
            risk_card(row)

    st.divider()
    left, right = st.columns([1.25, 1.0], gap="large")
    with left:
        st.markdown('<div class="section-label">Predicted event risks</div>', unsafe_allow_html=True)
        display = results[
            [
                "Outcome",
                "Endpoint",
                "Risk tier",
                "12-month risk",
                "24-month risk",
                "36-month risk",
                "Risk score",
            ]
        ].copy()
        for col in ["12-month risk", "24-month risk", "36-month risk"]:
            display[col] = display[col].map(format_percent)
        display["Risk score"] = display["Risk score"].map(lambda x: f"{x:.3f}")
        st.dataframe(display, hide_index=True, width="stretch")
    with right:
        st.markdown('<div class="section-label">Trajectory by horizon</div>', unsafe_allow_html=True)
        st.bar_chart(chart_data(results), y_label="Predicted event risk", height=330)

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Highest model-score stratum", RISK_STYLE[highest["Risk group"]]["label"])
    c2.metric("Training datasets", str(len(weights["training_datasets"])))
    c3.metric("External validation datasets", str(len(weights["external_validation_datasets"])))

    with st.expander("Model-score thresholds used for low/intermediate/high strata"):
        threshold_rows = []
        for _, row in results.iterrows():
            threshold_rows.append(
                {
                    "Outcome": row["Outcome"],
                    "Low if score <=": f"{row['Low/Intermediate threshold']:.3f}",
                    "Intermediate if score <=": f"{row['Intermediate/High threshold']:.3f}",
                    "High if score >": f"{row['Intermediate/High threshold']:.3f}",
                    "Patient score": f"{row['Risk score']:.3f}",
                }
            )
        st.dataframe(pd.DataFrame(threshold_rows), hide_index=True, width="stretch")

    with st.expander("Input profile"):
        profile = []
        for feature in features:
            meta = feature_meta[feature]
            value = input_row.loc[0, feature]
            display_meta = FEATURE_DISPLAY.get(feature, {})
            option_label = "Not available / unknown"
            if not pd.isna(value):
                option_label = display_meta.get("options", {}).get(float(value), value)
            profile.append(
                {
                    "Feature": display_meta.get("label", meta["label"]),
                    "Value": str(option_label),
                }
            )
        st.dataframe(pd.DataFrame(profile), hide_index=True, width="stretch")

    csv = results.drop(columns=["Outcome key"]).to_csv(index=False).encode("utf-8-sig")
    st.download_button("Download prediction CSV", data=csv, file_name="aml_allohct_risk_prediction.csv", mime="text/csv")

else:
    st.markdown('<div class="section-label">Model overview</div>', unsafe_allow_html=True)
    summary_rows = []
    for outcome, item in weights["training_summary"].items():
        summary_rows.append(
            {
                "Outcome": weights["outcome_labels"][outcome],
                "Endpoint": weights["outcome_descriptions"][outcome],
                "Development N": f"{item['n']:,}",
                "Events": f"{item['events']:,}",
                "Event rate": format_percent(item["event_rate"]),
            }
        )
    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, width="stretch")

    st.markdown("#### What the calculator returns")
    st.markdown(
        """
        - Model estimates at 12, 24, and 36 months for OS, relapse, DFS, and TRM/NRM.
        - Outcome-specific low/intermediate/high risk strata using development-cohort model-score tertiles.
        - Risk strata are model-derived research groups, not validated treatment recommendations or fixed probability cut-points.
        - Missing or unavailable categorical values are handled as explicit model inputs; such signals may partly reflect dataset, era, or testing-practice differences.
        """
    )

st.caption(
    "Displayed estimates are generated by a lightweight research calculator for online deployment. "
    "Relapse and TRM/NRM estimates should be interpreted as endpoint-specific research predictions."
)
