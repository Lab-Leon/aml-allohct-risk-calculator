from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


APP_ROOT = Path(__file__).resolve().parent
MODEL_DIR = APP_ROOT / "models"
COX_WEIGHTS_PATH = MODEL_DIR / "risk_calculator_weights.json"

MODEL_OPTIONS = {
    "CoxPH benchmark": "CoxPH benchmark",
}

OUTCOME_LABELS = {
    "OS": "Overall survival",
    "Relapse": "Relapse",
    "DFS": "Disease-free survival",
    "TRM_NRM": "TRM/NRM",
}

PROFILE_TITLES = {
    "OS": "OS event (death)",
    "Relapse": "Relapse (cause-specific)",
    "DFS": "DFS event (relapse/death)",
    "TRM_NRM": "TRM/NRM (cause-specific)",
}

OUTCOME_DESCRIPTIONS = {
    "OS": "Death from any cause",
    "Relapse": "Recorded relapse",
    "DFS": "Relapse or death",
    "TRM_NRM": "Non-relapse/treatment-related mortality",
}

ESTIMATE_LABELS = {
    "OS": "24-month model-estimated death risk",
    "Relapse": "24-month cause-specific relapse model estimate",
    "DFS": "24-month model-estimated relapse/death risk",
    "TRM_NRM": "24-month cause-specific TRM/NRM model estimate",
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

# The web interface intentionally presents the same clinically named, known
# category set used for the streamlined SHAP displays.  These values map
# directly to retained one-hot levels in the bundled CoxPH model. The UI does
# not expose raw source-code labels or missingness categories as clinical choices.
# Every categorical menu contains at most five options.
CLINICAL_INPUT_OPTIONS = {
    "sex": [
        ("1", "Male"),
        ("2", "Female"),
    ],
    "kps": [
        ("0.0", "KPS <80"),
        ("1.0", "KPS 80-90"),
        ("2.0", "KPS ≥90"),
    ],
    "hct_ci": [
        ("0.0", "HCT-CI 0"),
        ("1.0", "HCT-CI 1"),
        ("2.0", "HCT-CI 2"),
        ("3.0", "HCT-CI ≥3"),
    ],
    "cytogenetics": [
        ("2.0", "Favorable risk"),
        ("3.0", "Intermediate risk"),
        ("4.0", "Adverse risk"),
    ],
    "mrd": [
        ("0.0", "MRD negative"),
        ("1.0", "MRD positive"),
        ("2.0", "MRD indeterminate"),
    ],
    "donor_type": [
        ("1", "HLA-identical sibling donor"),
        ("2", "Other related or haploidentical donor"),
        ("3", "Matched unrelated donor"),
        ("4", "Mismatched unrelated donor"),
        ("5", "Cord blood donor"),
    ],
    "graft_type": [
        ("1", "Bone marrow graft"),
        ("22", "Peripheral blood stem-cell graft"),
        ("23", "Cord blood graft"),
    ],
    "conditioning_intensity": [
        ("1", "Myeloablative, TBI-based"),
        ("2", "Myeloablative, chemotherapy-based"),
        ("3", "Reduced-intensity/non-myeloablative"),
    ],
    "cmv": [
        ("0.0", "Recipient CMV negative"),
        ("1.0", "Recipient CMV positive"),
    ],
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
    .small-muted { color: #64748b; font-size: .82rem; }
    .section-label { font-weight: 720; color: #111827; margin: .25rem 0 .35rem 0; }
    div[data-testid="stMetric"] { background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 8px; padding: .55rem .7rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data(show_spinner=False)
def load_cox_weights() -> dict:
    return json.loads(COX_WEIGHTS_PATH.read_text(encoding="utf-8"))


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


def encode_cox_row(row: pd.Series, outcome_spec: dict) -> np.ndarray:
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


def predict_cox_outcome(row: pd.Series, weights: dict, outcome: str) -> dict:
    spec = weights["outcomes"][outcome]
    x = encode_cox_row(row, spec)
    score = float(np.dot(x, np.asarray(spec["coef"], dtype=float)))
    exp_score = math.exp(score)
    risks = [
        1.0 - float(spec["baseline_survival"][str(int(horizon))]) ** exp_score
        for horizon in weights["horizons"]
    ]
    group = risk_group(score, spec["risk_thresholds"])
    return {
        "Outcome key": outcome,
        "Outcome": OUTCOME_LABELS[outcome],
        "Endpoint": OUTCOME_DESCRIPTIONS[outcome],
        "Risk group": group,
        "Risk tier": RISK_STYLE[group]["label"],
        "12-month risk": risks[0],
        "24-month risk": risks[1],
        "36-month risk": risks[2],
        "Risk score": score,
        "Low/Intermediate threshold": spec["risk_thresholds"]["low_intermediate"],
        "Intermediate/High threshold": spec["risk_thresholds"]["intermediate_high"],
    }


def calculator_feature_metadata(weights: dict) -> dict:
    metadata = json.loads(json.dumps(weights["feature_metadata"]))
    for feature in weights["categorical_features"]:
        options = CLINICAL_INPUT_OPTIONS[feature]
        if len(options) > 5:
            raise ValueError(f"{feature} has more than five web-interface options")
        metadata[feature]["options"] = [
            {"value": token, "label": label}
            for token, label in options
        ]
    return metadata


def select_category(feature: str, meta: dict):
    display_meta = FEATURE_DISPLAY.get(feature, {})
    options = [
        {"label": item["label"], "value": item["value"]}
        for item in meta.get("options", [])
    ]
    selected = st.selectbox(
        display_meta.get("label", meta["label"]),
        options=list(range(len(options))),
        format_func=lambda i: options[i]["label"],
        index=1 if len(options) > 1 else 0,
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
    """Return CoxPH benchmark estimates from the bundled compact JSON weights."""
    input_row = row.iloc[0]
    return pd.DataFrame(
        [predict_cox_outcome(input_row, weights, outcome) for outcome in OUTCOME_LABELS]
    )


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


def endpoint_profile_figure(results: pd.DataFrame):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    horizons = np.asarray([12, 24, 36], dtype=float)
    columns = ["12-month risk", "24-month risk", "36-month risk"]
    outcomes = ["OS", "Relapse", "DFS", "TRM_NRM"]
    panels = ["a", "b", "c", "d"]

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.edgecolor": "#64748b",
            "axes.labelcolor": "#334155",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
        }
    ):
        fig, axes = plt.subplots(2, 2, figsize=(5.4, 3.65), dpi=160, sharex=True, sharey=True)
        fig.patch.set_facecolor("white")
        for ax, outcome, panel in zip(axes.ravel(), outcomes, panels):
            row = results.loc[results["Outcome key"] == outcome].iloc[0]
            estimates = row[columns].to_numpy(dtype=float)
            ax.plot(horizons, estimates, color="#1f5a94", marker="o", markersize=3.6, linewidth=1.5)
            for horizon, estimate in zip(horizons, estimates):
                ax.annotate(
                    f"{estimate:.0%}",
                    (horizon, estimate),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=5.6,
                    color="#334155",
                )
            ax.set_title(f"{panel}. {PROFILE_TITLES[outcome]}", loc="left", fontsize=7.2, fontweight="bold", pad=3)
            ax.set_xlim(9, 39)
            ax.set_ylim(0, 1.0)
            ax.set_xticks(horizons)
            ax.set_yticks([0.0, 0.5, 1.0])
            ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            ax.grid(axis="y", color="#e5e7eb", linewidth=0.6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(0.6)
            ax.spines["bottom"].set_linewidth(0.6)
            ax.tick_params(labelsize=6, width=0.6, length=2.5)
        fig.supylabel("Endpoint-specific model estimate", x=0.02, fontsize=6.5, color="#334155")
        fig.supxlabel("Months after transplantation", y=0.015, fontsize=6.5, color="#334155")
        fig.subplots_adjust(left=0.13, right=0.99, bottom=0.15, top=0.94, wspace=0.27, hspace=0.55)
    return fig


weights = load_cox_weights()
features = weights["features"]
feature_meta = calculator_feature_metadata(weights)

st.markdown('<div class="app-title">AML allo-HCT Risk Calculator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Research calculator for 12-, 24-, and 36-month post-transplant event risk and low/intermediate/high risk stratification after allogeneic HCT for AML.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="notice">
    Research prototype only. This lightweight distribution provides the CoxPH transportability benchmark for research
    communication, risk stratification, and recalibration studies. Prospective evaluation and target-site recalibration
    remain important before patient-level implementation.
    OS, DFS, relapse, and TRM/NRM were fitted independently; relapse and TRM/NRM are cause-specific model estimates,
    not native CIF probabilities, and are intended for endpoint-specific interpretation rather than arithmetic reconstruction of DFS.
    Input menus are restricted to the clinically named known categories displayed in the streamlined SHAP figures.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Patient and transplant inputs")
    selected_model = st.selectbox(
        "Prediction model",
        options=list(MODEL_OPTIONS),
        index=0,
        disabled=True,
        help=(
            "This minimal GitHub package bundles the compact CoxPH benchmark weights."
        ),
    )
    st.caption("The optional RSF research artifacts are intentionally excluded from this lightweight package.")
    st.caption("Use values known before or at transplantation. Each categorical menu is limited to the clinically named SHAP display categories (maximum five choices).")
    values = {}
    for feature in features:
        values[feature] = input_feature(feature, feature_meta[feature])
    st.caption("All displayed choices use clinically named categories.")
    calculate = st.button("Calculate risk", type="primary", width="stretch")

input_row = pd.DataFrame([{feature: values.get(feature, np.nan) for feature in features}])

if calculate:
    results = prediction_table(input_row, weights)
    highest = results.sort_values(["Risk group"], key=lambda s: s.map(risk_rank), ascending=False).iloc[0]

    st.markdown('<div class="section-label">Risk stratification summary</div>', unsafe_allow_html=True)
    st.caption(f"Current model: {selected_model}")
    st.caption(
        "Each stratum is outcome-specific and model-score based. OS and DFS are displayed as model-estimated event risks; "
        "relapse and TRM/NRM are cause-specific model estimates rather than native CIF probabilities."
    )
    cols = st.columns(4)
    for col, (_, row) in zip(cols, results.iterrows()):
        with col:
            risk_card(row)

    st.divider()
    left, right = st.columns([0.85, 1.55], gap="large")
    with left:
        st.markdown('<div class="section-label">Endpoint-specific estimates and strata</div>', unsafe_allow_html=True)
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
        st.divider()
        st.metric("Highest model-score stratum", RISK_STYLE[highest["Risk group"]]["label"])
        st.metric("Training datasets", str(len(weights["training_datasets"])))
        st.metric("External validation datasets", str(len(weights["external_validation_datasets"])))
    with right:
        st.markdown('<div class="section-label">Endpoint-specific estimates by horizon</div>', unsafe_allow_html=True)
        profile_figure = endpoint_profile_figure(results)
        st.pyplot(profile_figure, width="stretch", clear_figure=True)
        st.caption(
            "Separate endpoint-specific models are shown in independent panels; the profiles are not additive and do not "
            "form a single multistate probability system."
        )
    with st.expander("Risk score thresholds used for low/intermediate/high strata"):
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
                option_lookup = {
                    str(item["value"]): item["label"]
                    for item in meta.get("options", [])
                }
                option_label = option_lookup.get(str(value), value)
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
    st.caption(f"Selected model: {selected_model}")
    summary_rows = []
    for outcome, item in weights["training_summary"].items():
        summary_rows.append(
            {
                "Outcome": OUTCOME_LABELS[outcome],
                "Endpoint": OUTCOME_DESCRIPTIONS[outcome],
                "Development N": f"{item['n']:,}",
                "Events": f"{item['events']:,}",
                "Event rate": format_percent(item["event_rate"]),
            }
        )
    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, width="stretch")

    st.markdown("#### What the calculator returns")
    st.markdown(
        """
        - Model estimates at 12, 24, and 36 months for OS, relapse, DFS, and TRM/NRM; relapse and TRM/NRM are cause-specific model estimates rather than native CIF probabilities.
        - Outcome-specific low/intermediate/high risk strata using the selected model's development-cohort score tertiles.
        - Risk strata are model-derived research groups, not validated treatment recommendations or fixed probability cut-points.
        - Missing or unavailable categorical values are handled as explicit model inputs; such signals may partly reflect dataset, era, or testing-practice differences.
        """
    )

st.caption(
    "Relapse and TRM/NRM are cause-specific endpoint models. The displayed estimates are not native CIF estimates; "
    "the four endpoint-specific outputs are intended for separate interpretation rather than arithmetic reconstruction of DFS."
)
