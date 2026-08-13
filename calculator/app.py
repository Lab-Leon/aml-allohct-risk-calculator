from __future__ import annotations

import pandas as pd
import streamlit as st

from calculator.explorer_core import (
    CATEGORICAL,
    FEATURES,
    MODELS,
    OUTCOMES,
    SCHEMA,
    build_default_values,
    load_model_file,
    model_score,
    risk_rank,
    survival_estimates,
    validate_model_schema_contract,
)


DISPLAY_MODEL = {
    "CoxPH": "L2-penalized Cox",
    "Elastic-net Cox": "LASSO-Cox",
    "RSF": "Random survival forest (RSF)",
}
OUTCOME_LABELS = {
    "OS": "Overall survival (OS)",
    "Relapse": "Relapse",
    "DFS": "Disease-free survival (DFS)",
    "TRM_NRM": "Transplant-related / non-relapse mortality (TRM/NRM)",
}
STRATUM_COLOR = {
    "Low": "#2F6FAE",
    "Intermediate": "#707780",
    "High": "#B23A3A",
}


st.set_page_config(
    page_title="AML allo-HCT Research Model Explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --aml-blue: #1E5596;
        --aml-ink: #203047;
        --aml-muted: #617086;
        --aml-border: #DCE3EC;
        --aml-bg: #F5F7FA;
    }
    .stApp { background: var(--aml-bg); }
    .block-container { max-width: 1320px; padding-top: 1.8rem; padding-bottom: 3rem; }
    [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }
    h1, h2, h3, h4 { color: var(--aml-ink); letter-spacing: -0.015em; }
    .aml-hero {
        background: linear-gradient(120deg, #163F72 0%, #1E5596 62%, #2B6EA9 100%);
        color: white; border-radius: 16px; padding: 1.45rem 1.7rem 1.3rem;
        box-shadow: 0 8px 24px rgba(28, 66, 105, 0.16); margin-bottom: 0.85rem;
    }
    .aml-hero h1 { color: white; margin: 0 0 0.35rem; font-size: 2.05rem; }
    .aml-hero p { margin: 0; color: #E5EEF8; font-size: 0.98rem; }
    .aml-safety {
        background: #FFF6E5; border: 1px solid #E6C06B; border-left: 5px solid #C98C18;
        color: #5E4615; border-radius: 10px; padding: 0.72rem 0.9rem; margin: 0.65rem 0 1rem;
        font-size: 0.92rem;
    }
    .aml-card-title { margin-bottom: -0.55rem; }
    .aml-stratum {
        display: inline-block; color: white; border-radius: 999px; padding: 0.20rem 0.68rem;
        font-weight: 700; font-size: 0.84rem; margin-top: 0.1rem;
    }
    .aml-calibration-note {
        background: #FFF4F1; border-left: 4px solid #C55A43; border-radius: 7px;
        color: #6C3428; padding: 0.55rem 0.68rem; font-size: 0.82rem; line-height: 1.35;
        margin-top: 0.5rem;
    }
    .aml-ranking-note {
        background: #F0F5FA; border-left: 4px solid #557A9E; border-radius: 7px;
        color: #38536E; padding: 0.55rem 0.68rem; font-size: 0.82rem; line-height: 1.35;
        margin-top: 0.5rem;
    }
    button[kind="primaryFormSubmit"] {
        background: var(--aml-blue) !important; border-color: var(--aml-blue) !important;
        color: white !important;
    }
    button[kind="primaryFormSubmit"]:hover {
        background: #17477F !important; border-color: #17477F !important;
    }
    [data-testid="stMetricLabel"] { color: #617086; }
    [data-testid="stMetricValue"] { color: #203047; font-size: 1.55rem; }
    [data-testid="stForm"] {
        background: white; border-color: var(--aml-border); border-radius: 13px;
        box-shadow: 0 2px 10px rgba(32, 48, 71, 0.05);
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: white; border-color: var(--aml-border); border-radius: 13px;
        box-shadow: 0 2px 10px rgba(32, 48, 71, 0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


try:
    # Run this lightweight check directly. A Streamlit cache lock can survive a
    # cloud sleep/wake transition and block the first script delta, leaving only
    # the empty frontend shell visible until the app is rebooted.
    CONTRACT = validate_model_schema_contract()
except Exception:
    st.error(
        "The model explorer is temporarily unavailable. Please contact the study team."
    )
    st.stop()


st.markdown(
    """
    <div class="aml-hero">
      <h1>AML allo-HCT Research Model Explorer</h1>
      <p>Research-only exploration of development-selected survival and cause-specific risk models.</p>
    </div>
    <div class="aml-safety"><strong>Research use only.</strong> This interface is not validated or authorized for bedside treatment decisions, does not recommend treatment, and does not establish clinical utility.</div>
    """,
    unsafe_allow_html=True,
)


defaults = build_default_values()
with st.form("research_inputs", border=True):
    st.subheader("Patient-level research inputs")
    st.caption(
        "Dropdown choices are limited to categories represented in the development data. "
        "Missing/unknown categories remain available only where the fitted model was trained on them. "
        "Numeric age is restricted to 18–81 completed years, the integer range observed in development."
    )
    columns = st.columns(3)
    age_spec = SCHEMA["numeric"]["age"]
    with columns[0]:
        values = {
            "age": st.number_input(
                age_spec["label"],
                min_value=int(age_spec["min"]),
                max_value=int(age_spec["max"]),
                value=int(defaults["age"]),
                step=1,
            )
        }
    with columns[1]:
        model_name = st.selectbox(
            "Research model",
            MODELS,
            index=MODELS.index("RSF"),
            format_func=lambda value: DISPLAY_MODEL[value],
        )
    with columns[2]:
        st.markdown("**Output hierarchy**")
        st.caption("Development-score percentile and stratum are shown before any model-scale output.")

    for index, feature in enumerate(CATEGORICAL):
        specification = SCHEMA["categorical"][feature]
        with columns[index % 3]:
            values[feature] = st.selectbox(
                specification["label"],
                specification["options"],
                index=0,
                key=f"input_{feature}",
            )
    submitted = st.form_submit_button(
        "Update research estimates", type="primary", use_container_width=True
    )


low_support = []
for feature in CATEGORICAL:
    specification = SCHEMA["categorical"][feature]
    selected = values[feature]
    if specification["low_support_flags"][selected]:
        low_support.append(
            f"{specification['label']} = {selected} "
            f"(development n={specification['development_counts'][selected]:,})"
        )
if low_support:
    st.warning(
        "Low-support input under the prespecified rule (development n < 200 or prevalence < 1%): "
        + "; ".join(low_support)
        + ". Interpret the corresponding outputs cautiously."
    )


st.subheader("Research outputs")
st.caption(
    "Percentiles and strata are relative to the development-score distribution. "
    "They are not treatment thresholds and do not provide individual-prediction confidence intervals."
)


row = pd.DataFrame([values], columns=FEATURES)


def render_outcome_card(outcome: str) -> None:
    # Avoid a process-wide Streamlit resource lock. The compact models load in
    # well under a second and per-run loading is safer across cloud wake-ups.
    model = load_model_file(outcome, model_name, validate_encoder=True)
    value = model_score(model, row)
    percentile, group = risk_rank(value, outcome, model_name)
    with st.container(border=True):
        st.markdown(f"#### {OUTCOME_LABELS[outcome]}")
        primary_left, primary_right = st.columns([1.35, 1])
        with primary_left:
            st.metric("Development-score percentile", f"{percentile:.1f}")
        with primary_right:
            st.markdown("<div style='height:0.28rem'></div>", unsafe_allow_html=True)
            st.caption("Development-score stratum")
            st.markdown(
                f"<span class='aml-stratum' style='background:{STRATUM_COLOR[group]}'>{group}</span>",
                unsafe_allow_html=True,
            )

        if outcome in {"OS", "DFS"}:
            st.markdown("**Uncalibrated research estimate — survival probability**")
            estimates = survival_estimates(model, row)
            metric_columns = st.columns(3)
            for column, horizon, estimate in zip(metric_columns, (12, 24, 36), estimates):
                with column:
                    st.metric(f"{horizon} months", f"{estimate:.1%}")
            st.markdown(
                "<div class='aml-calibration-note'><strong>Calibration warning.</strong> External validation showed systematic miscalibration. These probabilities require setting-specific recalibration before any patient-level use.</div>",
                unsafe_allow_html=True,
            )
        else:
            score_left, score_right = st.columns([1, 1.4])
            with score_left:
                st.metric("Cause-specific risk score", f"{value:.3f}")
            with score_right:
                st.markdown(
                    "<div class='aml-ranking-note'><strong>Relative ranking only.</strong><br>This score is not an absolute event probability or cumulative-incidence estimate.</div>",
                    unsafe_allow_html=True,
                )


result_columns = st.columns(2)
for index, outcome in enumerate(OUTCOMES):
    with result_columns[index % 2]:
        render_outcome_card(outcome)


st.warning(
    "OS, DFS, relapse, and TRM/NRM were fitted independently. Do not add or combine "
    "relapse and TRM/NRM scores to reconstruct DFS."
)

with st.expander("How to interpret these research outputs"):
    st.markdown(
        """
        - OS and DFS show native model survival estimates only after the development-score percentile and stratum; the manuscript documents systematic external miscalibration.
        - Relapse and TRM/NRM show cause-specific ranking scores only. They are not probabilities and are not cumulative-incidence estimates.
        - No individual-prediction confidence interval is available. Model-behavior explanations are descriptive and not causal.
        - Local recalibration and prospective validation are required before any clinical implementation could be considered.
        """
    )
