import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.modeling import ModelRegistry

st.set_page_config(page_title="Diabetes Progression Screener", page_icon="🩺", layout="wide")

BLUE, RED, INK, MUTED, GRID = "#2a78d6", "#e34948", "#0b0b0b", "#52514e", "#e6e5e0"

MODELS = {
    "kNN, k = 11 (8 features)": "knn_regressor",
    "Linear regression (8 features)": "linear_regressor",
    "BMI only (degree 1 polynomial)": "bmi_regressor",
}

INPUTS = {
    "age": ("Age", "years", 1.0),
    "bmi": ("Body mass index", "kg/m²", 0.1),
    "bp": ("Average blood pressure", "mm Hg", 1.0),
    "hdl": ("HDL cholesterol", "mg/dL", 1.0),
    "tch": ("Total cholesterol / HDL", "ratio", 0.1),
    "ltg": ("Serum triglycerides (log)", "log scale", 0.01),
    "glu": ("Blood sugar", "mg/dL", 1.0),
}


@st.cache_resource
def load_assets():
    registry = ModelRegistry("models")
    models = {key: registry.load(key) for key in [*MODELS.values(), "risk_classifier"]}
    return models, registry.read_metadata()


def number_field(column, feature, meta):
    label, unit, step = INPUTS[feature]
    r = meta["ranges"][feature]
    decimals = 2 if step < 0.1 else 1 if step < 1 else 0
    return column.number_input(
        f"{label} ({unit})",
        value=round(r["median"], decimals),
        step=step,
        format=f"%.{decimals}f",
        help=f"Training range {r['min']:g} to {r['max']:g}",
    )


def distribution_chart(targets, score, threshold):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=targets, nbinsx=30, marker=dict(color="#b7d3f6", line=dict(color="white", width=1.5)),
                               hovertemplate="Score %{x}<br>Patients %{y}<extra></extra>"))
    fig.add_vline(x=threshold, line=dict(color=RED, width=2, dash="dash"),
                  annotation_text="high-risk cut-off", annotation_position="top left")
    fig.add_vline(x=score, line=dict(color=INK, width=3),
                  annotation_text=f"this patient: {score:.0f}", annotation_position="top right")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10), showlegend=False,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(title="Progression score of training patients", gridcolor=GRID),
                      yaxis=dict(title="Patients", gridcolor=GRID))
    return fig


def risk_gauge(probability):
    color = RED if probability >= 0.5 else BLUE
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=probability * 100, number=dict(suffix="%", font=dict(size=44, color=INK)),
        gauge=dict(axis=dict(range=[0, 100], tickcolor=MUTED), bar=dict(color=color, thickness=0.35),
                   bgcolor="#f0efec", borderwidth=0,
                   threshold=dict(line=dict(color=INK, width=3), thickness=0.8, value=50)),
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=0), paper_bgcolor="rgba(0,0,0,0)")
    return fig


models, meta = load_assets()
features = meta["features"]
threshold = meta["risk_threshold"]

st.title("Diabetes progression screener")
st.caption(
    "Estimates disease progression one year after baseline and the chance that a patient lands in the "
    "high-risk group. A screening aid for clinicians, not a diagnosis."
)

with st.sidebar:
    st.header("Model")
    choice = st.radio("Score model", list(MODELS), index=0)
    key = MODELS[choice]
    scores = meta["test_scores"][key]
    st.markdown("**Test set performance**")
    a, b, c = st.columns(3)
    a.metric("R²", f"{scores['R2']:.2f}")
    b.metric("MAE", f"{scores['MAE']:.0f}")
    c.metric("MAPE", f"{scores['MAPE_%']:.0f}%")
    clf = meta["test_scores"]["risk_classifier"]
    st.markdown("**Risk classifier (balanced logistic regression)**")
    a, b = st.columns(2)
    a.metric("Recall", f"{clf['recall']:.2f}")
    b.metric("ROC-AUC", f"{clf['ROC_AUC']:.2f}")
    st.caption(f"High risk means a predicted score of {threshold:.0f} or more (top 25% of training patients).")

with st.form("patient"):
    left, right = st.columns(2)
    left.subheader("Patient profile")
    values = {"age": number_field(left, "age", meta)}
    values["sex"] = left.radio("Sex (dataset code)", [1, 2], horizontal=True,
                               help="The source data codes sex as 1 or 2 without naming the groups.")
    for feature in ["bmi", "bp"]:
        values[feature] = number_field(left, feature, meta)
    right.subheader("Blood work")
    for feature in ["hdl", "tch", "ltg", "glu"]:
        values[feature] = number_field(right, feature, meta)
    submitted = st.form_submit_button("Predict", type="primary")

if submitted:
    patient = pd.DataFrame([values])[features]
    model_input = patient[["bmi"]] if key == "bmi_regressor" else patient
    score = float(models[key].predict(model_input)[0])
    probability = float(models["risk_classifier"].predict_proba(patient)[0, 1])
    flagged = probability >= 0.5

    outside = [
        INPUTS[f][0] for f in INPUTS
        if not meta["ranges"][f]["min"] <= values[f] <= meta["ranges"][f]["max"]
    ]
    if outside:
        st.warning(
            "Outside the range seen in training: " + ", ".join(outside)
            + ". Predictions here are extrapolations and less reliable, especially for kNN."
        )

    st.divider()
    col_score, col_risk = st.columns([3, 2])
    with col_score:
        st.subheader("Predicted progression")
        st.metric("Score one year after baseline", f"{score:.0f}",
                  delta=f"{score - threshold:+.0f} vs high-risk cut-off", delta_color="inverse")
        st.plotly_chart(distribution_chart(meta["training_targets"], score, threshold))
    with col_risk:
        st.subheader("Probability of high risk")
        st.plotly_chart(risk_gauge(probability))
        if flagged:
            st.error("⚠ Flag for follow-up: the classifier places this patient in the high-risk group.")
        else:
            st.success("✓ Not flagged: below the high-risk threshold.")

    if (score >= threshold) != flagged:
        st.info(
            "The score model and the risk classifier disagree. The classifier was tuned to catch high-risk "
            f"patients (test recall {clf['recall']:.2f}), while score models pull extreme patients toward the average, "
            "so the risk flag is the safer guide for screening."
        )

with st.expander("How the models were built"):
    st.markdown(
        "Trained on 331 patients from the scikit-learn diabetes dataset (Efron et al., 2004), validated on 44 "
        "and tested on 67. Total cholesterol and LDL were dropped as redundant with the other cholesterol "
        "measures. Full analysis in `diabetes_progression.ipynb`."
    )
