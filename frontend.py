import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import pickle
import streamlit as st

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:  # app still works without plotly
    HAS_PLOTLY = False

# ===============================
# Page configuration
# ===============================
st.set_page_config(
    page_title="Multiple Disease Risk Screening",
    page_icon="🩺",
    layout="wide",
)

# ===============================
# Theme
# ===============================
INK, PANEL, LINE = "#0E1A22", "#15262F", "#254250"
TEXT, MUTED, ACCENT = "#E6EEF0", "#8FA6AF", "#3CC8B4"
LOW, MID, HIGH = "#5BC98A", "#F2B84B", "#EF6F61"

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Figtree:wght@400;500;600&display=swap');

html, body, [class*="css"], .stApp {{ font-family: 'Figtree', sans-serif; }}
.stApp {{ background: {INK}; color: {TEXT}; }}
header[data-testid="stHeader"] {{ background: transparent; }}
section[data-testid="stSidebar"] {{ background: {PANEL}; border-right: 1px solid {LINE}; }}

.hero {{ padding: 8px 0 18px 0; }}
.hero h1 {{ font-family: 'Fraunces', serif; font-size: 2.9rem; font-weight: 700;
           letter-spacing: -0.5px; margin: 0; color: {TEXT}; }}
.hero p  {{ color: {MUTED}; font-size: 1.1rem; margin: 6px 0 0 0; max-width: 640px; }}

.stat {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 10px; padding: 14px 18px; }}
.stat .v {{ font-family: 'Fraunces', serif; font-size: 1.6rem; color: {TEXT}; }}
.stat .l {{ color: {MUTED}; font-size: 0.85rem; }}

.result {{ border-radius: 12px; padding: 22px 26px; background: {PANEL};
          border: 1px solid {LINE}; border-left: 6px solid var(--tone); }}
.result .band {{ font-family: 'Fraunces', serif; font-size: 1.9rem; color: var(--tone); margin: 0; }}
.result .msg  {{ color: {MUTED}; margin-top: 6px; }}

div[data-testid="stForm"] {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 12px; padding: 22px; }}
div.stButton > button, div[data-testid="stFormSubmitButton"] > button {{
    background: {ACCENT}; color: {INK}; font-weight: 600; border: 0; border-radius: 8px; padding: 0.55rem 1.6rem; }}
div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {{ background: #62dcca; color: {INK}; }}
button[role="tab"] {{ font-weight: 500; }}
.disclaimer {{ color: {MUTED}; font-size: 0.85rem; border-top: 1px solid {LINE}; padding-top: 12px; margin-top: 28px; }}
</style>
""",
    unsafe_allow_html=True,
)

# ===============================
# Disease catalogue (schema-driven inputs)
#   ("num", key, label, min, max, default, step, help)
#   ("sel", key, label, {option: value}, help)   -> first option is the default
# Insertion order of keys is preserved when the DataFrame is built.
# ===============================
DISEASES = {
    "breast_cancer": {
        "label": "Breast cancer", "icon": "🎗️",
        "blurb": "Cell-nucleus measurements from a fine-needle aspirate.",
        "fields": [
            ("num", "radius_mean", "Radius (mean)", 5.0, 30.0, 14.13, 0.01, None),
            ("num", "texture_mean", "Texture (mean)", 5.0, 45.0, 19.29, 0.01, None),
            ("num", "perimeter_mean", "Perimeter (mean)", 40.0, 200.0, 91.97, 0.01, None),
            ("num", "area_mean", "Area (mean)", 140.0, 2600.0, 654.9, 0.1, None),
            ("num", "smoothness_mean", "Smoothness (mean)", 0.04, 0.2, 0.0964, 0.0001, None),
            ("num", "compactness_mean", "Compactness (mean)", 0.01, 0.4, 0.1043, 0.0001, None),
            ("num", "concavity_mean", "Concavity (mean)", 0.0, 0.5, 0.0888, 0.0001, None),
            ("num", "concave points_mean", "Concave points (mean)", 0.0, 0.25, 0.0489, 0.0001, None),
            ("num", "symmetry_mean", "Symmetry (mean)", 0.1, 0.35, 0.1812, 0.0001, None),
            ("num", "fractal_dimension_mean", "Fractal dimension (mean)", 0.04, 0.1, 0.0628, 0.0001, None),
            ("num", "radius_se", "Radius (SE)", 0.1, 3.0, 0.405, 0.001, None),
            ("num", "texture_se", "Texture (SE)", 0.3, 5.0, 1.217, 0.001, None),
            ("num", "perimeter_se", "Perimeter (SE)", 0.7, 22.0, 2.866, 0.001, None),
            ("num", "area_se", "Area (SE)", 6.0, 550.0, 40.34, 0.01, None),
            ("num", "smoothness_se", "Smoothness (SE)", 0.001, 0.04, 0.00704, 0.0001, None),
            ("num", "compactness_se", "Compactness (SE)", 0.001, 0.15, 0.0255, 0.0001, None),
            ("num", "concavity_se", "Concavity (SE)", 0.0, 0.4, 0.0319, 0.0001, None),
            ("num", "concave points_se", "Concave points (SE)", 0.0, 0.06, 0.0118, 0.0001, None),
            ("num", "symmetry_se", "Symmetry (SE)", 0.007, 0.08, 0.0205, 0.0001, None),
            ("num", "fractal_dimension_se", "Fractal dimension (SE)", 0.0008, 0.03, 0.00379, 1e-05, None),
            ("num", "radius_worst", "Radius (worst)", 7.0, 40.0, 16.27, 0.01, None),
            ("num", "texture_worst", "Texture (worst)", 10.0, 55.0, 25.68, 0.01, None),
            ("num", "perimeter_worst", "Perimeter (worst)", 45.0, 260.0, 107.26, 0.01, None),
            ("num", "area_worst", "Area (worst)", 180.0, 4300.0, 880.6, 0.1, None),
            ("num", "smoothness_worst", "Smoothness (worst)", 0.05, 0.25, 0.1324, 0.0001, None),
            ("num", "compactness_worst", "Compactness (worst)", 0.02, 1.1, 0.2543, 0.0001, None),
            ("num", "concavity_worst", "Concavity (worst)", 0.0, 1.3, 0.2722, 0.0001, None),
            ("num", "concave points_worst", "Concave points (worst)", 0.0, 0.3, 0.1146, 0.0001, None),
            ("num", "symmetry_worst", "Symmetry (worst)", 0.15, 0.7, 0.2901, 0.0001, None),
            ("num", "fractal_dimension_worst", "Fractal dimension (worst)", 0.05, 0.21, 0.0839, 0.0001, None),
        ],
    },
    "heart": {
        "label": "Heart disease", "icon": "❤️",
        "blurb": "Clinical and exercise-test indicators of coronary risk.",
        "fields": [
            ("num", "age", "Age (years)", 1, 120, 55, 1, None),
            ("sel", "sex", "Sex", {"Male": 1, "Female": 0}, None),
            ("sel", "cp", "Chest pain type",
             {"Typical angina": 0, "Atypical angina": 1, "Non-anginal pain": 2, "Asymptomatic": 3}, None),
            ("num", "trestbps", "Resting blood pressure (mm Hg)", 80, 220, 130, 1, None),
            ("num", "chol", "Cholesterol (mg/dl)", 100, 600, 240, 1, None),
            ("sel", "fbs", "Fasting blood sugar > 120 mg/dl", {"No": 0, "Yes": 1}, None),
            ("sel", "restecg", "Resting ECG",
             {"Normal": 0, "ST-T abnormality": 1, "Left ventricular hypertrophy": 2}, None),
            ("num", "thalach", "Max heart rate (bpm)", 60, 220, 150, 1, None),
            ("sel", "exang", "Exercise-induced angina", {"No": 0, "Yes": 1}, None),
            ("num", "oldpeak", "ST depression", 0.0, 7.0, 1.0, 0.1, "Exercise-induced ST depression vs. rest"),
            ("sel", "slope", "ST slope", {"Upsloping": 0, "Flat": 1, "Downsloping": 2}, None),
            ("num", "ca", "Major vessels coloured (0–4)", 0, 4, 0, 1, None),
            ("sel", "thal", "Thalassemia",
             {"Normal": 2, "Fixed defect": 1, "Reversible defect": 3, "Unknown": 0},
             "Encoding used by the heart.csv training data"),
        ],
    },
    "kidney": {
        "label": "Chronic kidney disease", "icon": "🫘",
        "blurb": "Blood, urine and clinical markers of kidney function.",
        "fields": [
            ("num", "age", "Age (years)", 1, 100, 55, 1, None),
            ("num", "bp", "Blood pressure (mm Hg)", 40, 200, 80, 1, None),
            ("num", "sg", "Specific gravity", 1.005, 1.025, 1.02, 0.005, "Urine specific gravity"),
            ("num", "al", "Albumin (0–5)", 0, 5, 0, 1, "Urine albumin grade"),
            ("num", "su", "Sugar (0–5)", 0, 5, 0, 1, "Urine sugar grade"),
            ("sel", "rbc", "Red blood cells", {"Normal": 1, "Abnormal": 0}, None),
            ("sel", "pc", "Pus cells", {"Normal": 1, "Abnormal": 0}, None),
            ("sel", "pcc", "Pus cell clumps", {"Not present": 0, "Present": 1}, None),
            ("sel", "ba", "Bacteria", {"Not present": 0, "Present": 1}, None),
            ("num", "bgr", "Blood glucose, random (mg/dl)", 20, 500, 120, 1, None),
            ("num", "bu", "Blood urea (mg/dl)", 1.0, 400.0, 42.0, 1.0, None),
            ("num", "sc", "Serum creatinine (mg/dl)", 0.1, 80.0, 1.4, 0.1, None),
            ("num", "sod", "Sodium (mEq/L)", 4.0, 170.0, 138.0, 0.5, None),
            ("num", "pot", "Potassium (mEq/L)", 2.0, 50.0, 4.4, 0.1, None),
            ("num", "hemo", "Hemoglobin (g/dl)", 3.0, 18.0, 12.6, 0.1, None),
            ("num", "pcv", "Packed cell volume", 9, 60, 40, 1, None),
            ("num", "wc", "White blood cell count (cells/cumm)", 2000, 27000, 8000, 100, None),
            ("num", "rc", "Red blood cell count (millions/cmm)", 2.0, 8.0, 4.8, 0.1, None),
            ("sel", "htn", "Hypertension", {"No": 0, "Yes": 1}, None),
            ("sel", "dm", "Diabetes mellitus", {"No": 0, "Yes": 1}, None),
            ("sel", "cad", "Coronary artery disease", {"No": 0, "Yes": 1}, None),
            ("sel", "appet", "Appetite", {"Good": 0, "Poor": 1}, None),
            ("sel", "pe", "Pedal edema", {"No": 0, "Yes": 1}, None),
            ("sel", "ane", "Anemia", {"No": 0, "Yes": 1}, None),
        ],
    },
    "diabetes": {
        "label": "Diabetes", "icon": "🩸",
        "blurb": "Metabolic indicators from the Pima Indians dataset.",
        "fields": [
            ("num", "Pregnancies", "Pregnancies", 0, 20, 3, 1, None),
            ("num", "Glucose", "Glucose (mg/dl)", 0.0, 300.0, 120.0, 1.0, "Enter 0 if unknown: the training average is used"),
            ("num", "BloodPressure", "Blood pressure (mm Hg)", 0.0, 200.0, 72.0, 1.0, "Enter 0 if unknown: the training average is used"),
            ("num", "SkinThickness", "Skin thickness (mm)", 0.0, 100.0, 23.0, 1.0, "Enter 0 if unknown: the training average is used"),
            ("num", "Insulin", "Insulin (µU/ml)", 0.0, 900.0, 30.0, 1.0, "Enter 0 if unknown: the training average is used"),
            ("num", "BMI", "BMI", 0.0, 70.0, 32.0, 0.1, "Enter 0 if unknown: the training average is used"),
            ("num", "DiabetesPedigreeFunction", "Diabetes pedigree function", 0.0, 2.5, 0.37, 0.01, "Family-history score"),
            ("num", "Age", "Age (years)", 1, 120, 29, 1, None),
        ],
    },
}

# Zeros in these diabetes columns were treated as missing and mean-filled during training
DIABETES_MEANS = {"Glucose": 121.69, "BloodPressure": 72.41, "SkinThickness": 29.15,
                  "Insulin": 155.55, "BMI": 32.46}

# Long column names used in the kidney notebook (in case the preprocessor was fitted on them)
KIDNEY_ALIASES = {
    "bp": "blood_pressure", "sg": "specific_gravity", "al": "albumin", "su": "sugar",
    "rbc": "red_blood_cells", "pc": "pus_cell", "pcc": "pus_cell_clumps", "ba": "bacteria",
    "bgr": "blood_glucose_random", "bu": "blood_urea", "sc": "serum_creatinine",
    "sod": "sodium", "pot": "potassium", "hemo": "haemoglobin", "pcv": "packed_cell_volume",
    "wc": "white_blood_cell_count", "rc": "red_blood_cell_count", "htn": "hypertension",
    "dm": "diabetes_mellitus", "cad": "coronary_artery_disease", "appet": "appetite",
    "pe": "peda_edema", "ane": "aanemia",
}

# ===============================
# Helpers
# ===============================
def model_path(ds):
    return f"Models/{ds}/model.pkl"


def preprocessor_path(ds):
    return f"artifacts/{ds}/preprocessor.pkl"


@st.cache_resource(show_spinner=False)
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def prepare_frame(df, preprocessor):
    """Match column names/order to what the fitted preprocessor expects."""
    names = getattr(preprocessor, "feature_names_in_", None)
    if names is None:
        return df
    names = list(names)
    norm = lambda s: re.sub(r"[\s_]+", "", str(s).lower())
    by_norm = {norm(n): n for n in names}
    rename = {}
    for c in df.columns:
        if c in names:
            continue
        if KIDNEY_ALIASES.get(c) in names:
            rename[c] = KIDNEY_ALIASES[c]
        elif norm(c) in by_norm:
            rename[c] = by_norm[norm(c)]
    df = df.rename(columns=rename)
    # Row-identifier columns (e.g. "id" left over from the CSV) carry no medical
    # information, so a placeholder value is used if the preprocessor still expects one.
    for n in names:
        if n not in df.columns and str(n).strip().lower() in {"id", "unnamed: 0", "index"}:
            df[n] = 0
    missing = [n for n in names if n not in df.columns]
    if missing:
        raise ValueError(f"The preprocessor expects columns the form doesn't collect: {missing}")
    return df[names]


def risk_band(p):
    if p < 0.35:
        return "Lower risk", LOW, "The model sees few indicators of this condition in the values entered."
    if p < 0.65:
        return "Moderate risk", MID, "The model is uncertain. Consider a clinical follow-up to clarify."
    return "Higher risk", HIGH, "The model sees strong indicators of this condition. Please consult a clinician."


def gauge(p, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(p * 100, 1),
        number={"suffix": "%", "font": {"size": 40, "color": TEXT}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": MUTED},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": PANEL, "borderwidth": 0,
            "steps": [
                {"range": [0, 35], "color": "#1c3a33"},
                {"range": [35, 65], "color": "#3b3322"},
                {"range": [65, 100], "color": "#3d2523"},
            ],
        },
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", font={"color": TEXT})
    return fig


def render_inputs(ds):
    """Build widgets from the schema; returns (values for model, readable values)."""
    values, readable = {}, {}
    cols = st.columns(3)
    for i, f in enumerate(DISEASES[ds]["fields"]):
        col, kind, key, label = cols[i % 3], f[0], f[1], f[2]
        wkey = f"{ds}_{key}"
        if kind == "num":
            _, _, _, lo, hi, default, step, help_ = f
            fmt = "%d" if isinstance(default, int) else ("%.5f" if step < 0.0001 else "%.4f" if step < 0.001 else "%.3f" if step < 0.01 else "%.2f" if step < 0.1 else "%.1f")
            val = col.number_input(label, min_value=lo, max_value=hi, value=default,
                                   step=step, format=fmt, help=help_, key=wkey)
            values[key], readable[label] = val, val
        else:
            _, _, _, mapping, help_ = f
            choice = col.selectbox(label, list(mapping.keys()), help=help_, key=wkey)
            values[key], readable[label] = mapping[choice], choice
    return values, readable


def feature_importance(model, preprocessor):
    """Return a Series of importances if the model exposes them, else None."""
    try:
        if hasattr(model, "feature_importances_"):
            imp = np.asarray(model.feature_importances_)
        elif hasattr(model, "coef_"):
            imp = np.abs(np.asarray(model.coef_)).ravel()
        else:
            return None
        names = list(preprocessor.get_feature_names_out())
        names = [n.split("__")[-1] for n in names]
        if len(names) != len(imp):
            return None
        return pd.Series(imp, index=names).sort_values(ascending=False).head(12)[::-1]
    except Exception:
        return None


# ===============================
# Session state
# ===============================
st.session_state.setdefault("history", [])
st.session_state.setdefault("last", None)

# ===============================
# Sidebar
# ===============================
with st.sidebar:
    st.markdown("### 🩺 Screening panel")
    dataset = st.selectbox(
        "Condition",
        list(DISEASES.keys()),
        format_func=lambda k: f"{DISEASES[k]['icon']}  {DISEASES[k]['label']}",
    )
    st.caption(DISEASES[dataset]["blurb"])

    ready = os.path.exists(model_path(dataset)) and os.path.exists(preprocessor_path(dataset))
    if ready:
        st.success("Model and preprocessor found", icon="✅")
    else:
        st.error("Model files missing", icon="⚠️")
        st.caption(f"Expected `{model_path(dataset)}` and `{preprocessor_path(dataset)}`")

    st.divider()
    st.metric("Assessments this session", len(st.session_state.history))
    if st.button("Clear history", use_container_width=True):
        st.session_state.history = []
        st.session_state.last = None
        st.rerun()

# ===============================
# Hero
# ===============================
st.markdown(
    f"""
<div class="hero">
  <h1>Multiple Disease Risk Screening</h1>
  <p>Enter a patient's values, get a model-based risk estimate for
  {DISEASES[dataset]['label'].lower()}, and see how confident the model is.</p>
</div>
""",
    unsafe_allow_html=True,
)

tab_assess, tab_insights, tab_history, tab_about = st.tabs(
    ["Assessment", "Model insights", "Session history", "About"]
)

# ===============================
# Assessment tab
# ===============================
with tab_assess:
    st.subheader(f"{DISEASES[dataset]['icon']} {DISEASES[dataset]['label']}")
    with st.form("prediction_form"):
        input_data, readable = render_inputs(dataset)
        submit = st.form_submit_button("Run assessment")

    if submit:
        if not ready:
            st.error("Model files are missing, so the assessment can't run. "
                     "Check the paths shown in the sidebar.")
        else:
            try:
                with st.spinner("Analysing values..."):
                    model = load_pickle(model_path(dataset))
                    preprocessor = load_pickle(preprocessor_path(dataset))
                    model_input = dict(input_data)
                    if dataset == "diabetes":
                        for k, mean in DIABETES_MEANS.items():
                            if model_input[k] == 0:
                                model_input[k] = mean
                    frame = prepare_frame(pd.DataFrame([model_input]), preprocessor)
                    transformed = preprocessor.transform(frame)
                    prediction = int(model.predict(transformed)[0])

                    prob = None
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(transformed)[0]
                        classes = list(getattr(model, "classes_", [0, 1]))
                        prob = float(probs[classes.index(1)] if 1 in classes else probs[-1])

                st.session_state.last = {
                    "dataset": dataset, "prediction": prediction, "prob": prob,
                    "inputs": readable, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                st.session_state.history.append({
                    "Time": st.session_state.last["time"],
                    "Condition": DISEASES[dataset]["label"],
                    "Result": "Positive" if prediction == 1 else "Negative",
                    "Risk probability": None if prob is None else round(prob * 100, 1),
                })
            except Exception as e:
                st.error(f"The assessment failed: {e}")

    last = st.session_state.last
    if last and last["dataset"] == dataset:
        p = last["prob"] if last["prob"] is not None else float(last["prediction"])
        band, tone, msg = risk_band(p)

        left, right = st.columns([3, 2], gap="large")
        with left:
            verdict = "Model flags this condition" if last["prediction"] == 1 else "Model does not flag this condition"
            st.markdown(
                f"""<div class="result" style="--tone:{tone}">
                <p class="band">{band}</p>
                <div style="font-weight:600;margin-top:4px">{verdict}</div>
                <div class="msg">{msg}</div></div>""",
                unsafe_allow_html=True,
            )
            with st.expander("Values used in this assessment"):
                st.dataframe(pd.DataFrame({"Value": last["inputs"]}).astype(str), use_container_width=True)

            report = pd.DataFrame({"Field": list(last["inputs"].keys()) + ["Result", "Risk probability (%)", "Time"],
                                   "Value": [str(v) for v in last["inputs"].values()] + [
                                       "Positive" if last["prediction"] == 1 else "Negative",
                                       "n/a" if last["prob"] is None else round(last["prob"] * 100, 1),
                                       last["time"]]})
            st.download_button("Download report (CSV)", report.to_csv(index=False).encode(),
                               file_name=f"{dataset}_report.csv", mime="text/csv")
        with right:
            if last["prob"] is None:
                st.info("This model doesn't provide probabilities, so no confidence gauge is shown.")
            elif HAS_PLOTLY:
                st.plotly_chart(gauge(last["prob"], tone), use_container_width=True)
                st.caption("Estimated probability that the condition is present")
            else:
                st.metric("Risk probability", f"{last['prob']*100:.1f}%")
                st.progress(int(last["prob"] * 100))

# ===============================
# Insights tab
# ===============================
with tab_insights:
    if not ready:
        st.info("Add the model files for this condition to see insights.")
    else:
        try:
            model = load_pickle(model_path(dataset))
            preprocessor = load_pickle(preprocessor_path(dataset))
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"<div class='stat'><div class='v'>{type(model).__name__}</div>"
                        f"<div class='l'>Algorithm</div></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='stat'><div class='v'>{len(DISEASES[dataset]['fields'])}</div>"
                        f"<div class='l'>Input features</div></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='stat'><div class='v'>{'Yes' if hasattr(model, 'predict_proba') else 'No'}</div>"
                        f"<div class='l'>Probability output</div></div>", unsafe_allow_html=True)

            names = getattr(preprocessor, "feature_names_in_", None)
            if names is not None:
                form_cols = {f[1] for f in DISEASES[dataset]["fields"]}
                try:
                    prepare_frame(pd.DataFrame([{k: 0 for k in form_cols}]), preprocessor)
                    st.success(f"Form inputs match the preprocessor ({len(names)} columns).", icon="✅")
                except ValueError as err:
                    st.warning(str(err), icon="⚠️")

            st.markdown("#### What the model relies on most")
            imp = feature_importance(model, preprocessor)
            if imp is None:
                st.caption("This model type doesn't expose feature importances.")
            elif HAS_PLOTLY:
                fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=ACCENT))
                fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0),
                                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  font={"color": TEXT}, xaxis=dict(gridcolor=LINE))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(imp)
        except Exception as e:
            st.error(f"Couldn't load model details: {e}")

# ===============================
# History tab
# ===============================
with tab_history:
    if st.session_state.history:
        hist = pd.DataFrame(st.session_state.history)
        st.dataframe(hist, use_container_width=True, hide_index=True)
        st.download_button("Download history (CSV)", hist.to_csv(index=False).encode(),
                           file_name="assessment_history.csv", mime="text/csv")
    else:
        st.info("No assessments yet. Run one in the Assessment tab and it will appear here.")

# ===============================
# About tab
# ===============================
with tab_about:
    st.markdown(
        """
**How it works.** Each condition has its own trained model and preprocessor. Your inputs are
transformed with the saved preprocessor, then scored by the model. When the model supports it,
the score is shown as a probability and grouped into lower (under 35%), moderate (35–65%) and higher (over 65%) risk.

**Good to know**
- Results come from statistical models trained on public datasets and may not reflect your patient population.
- Values are only as good as the measurements entered. Check units before running an assessment.
- Session history is kept in memory and is cleared when the browser tab is closed.
"""
    )

st.markdown(
    "<div class='disclaimer'>For educational and screening support only. This tool does not diagnose any condition "
    "and is not a substitute for professional medical advice.</div>",
    unsafe_allow_html=True,
)