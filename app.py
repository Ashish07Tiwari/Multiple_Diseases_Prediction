import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# ===============================
# Page Configuration
# ===============================
st.set_page_config(
    page_title="Multiple Disease Prediction System",
    page_icon="🩺",
    layout="wide"
)

# ===============================
# Custom CSS
# ===============================
st.markdown("""
<style>
    /* Dark Background */
    .stApp {
        background-color: #121212;
        color: #ffffff;
    }

    /* Main title - changed to yellow */
    .title {font-size: 48px; font-weight: 700; color:#FFD700; text-align:center;}
    .sub {font-size:22px; color:#b0bec5; text-align:center; margin-bottom:30px;}

    /* Card style */
    .card {
        padding: 30px;
        border-radius: 15px;
        background-color: #1e1e1e;
        box-shadow: 0px 8px 20px rgba(0,0,0,0.6);
        margin-bottom: 20px;
        color: #ffffff;
    }

    /* Success / Danger messages */
    .success {font-size:24px; color: #4caf50; font-weight: bold; text-align:center;}
    .danger {font-size:24px; color: #f44336; font-weight: bold; text-align:center;}

    /* Sidebar */
    .sidebar .sidebar-content {
        background-color: #1e1e1e;
        color: #ffffff;
    }

    .stSelectbox, .stNumberInput {
        background-color: #2c2c2c;
        color: #ffffff;
    }

</style>
""", unsafe_allow_html=True)


# ===============================
# Load model and preprocessor
# ===============================
def load_model(dataset):
    model_path = f"Models/{dataset}/model.pkl"
    with open(model_path, "rb") as f:
        return pickle.load(f)

def load_preprocessor(dataset):
    preprocessor_path = f"artifacts/{dataset}/preprocessor.pkl"
    with open(preprocessor_path, "rb") as f:
        return pickle.load(f)

# ===============================
# Sidebar
# ===============================
st.sidebar.title("🩺 Disease Selection")
dataset = st.sidebar.selectbox(
    "Choose Disease",
    ("breast_cancer", "heart", "kidney", "diabetes")
)
st.sidebar.info("Fill patient details and click **Predict**")

# ===============================
# Title
# ===============================
st.markdown("<div class='title'>Multiple Disease Prediction System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>AI-powered medical risk prediction</div><br>", unsafe_allow_html=True)

# ===============================
# Input Form
# ===============================
with st.form("prediction_form"):
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    input_data = {}

    # -------------------------------
    # Breast Cancer Inputs
    # -------------------------------
    if dataset == "breast_cancer":
        cols = st.columns(3)
        features = [
            "radius_mean", "texture_mean", "perimeter_mean",
            "area_mean", "smoothness_mean", "compactness_mean",
            "concavity_mean", "concave_points_mean", "symmetry_mean",
            "fractal_dimension_mean"
        ]
        for i, feature in enumerate(features):
            input_data[feature] = cols[i % 3].number_input(
                feature.replace("_", " ").title(), min_value=0.0
            )

    # -------------------------------
    # Heart Disease Inputs
    # -------------------------------
    elif dataset == "heart":
        cols = st.columns(3)
        input_data["age"] = cols[0].number_input("Age", 1, 120)
        input_data["trestbps"] = cols[1].number_input("Resting Blood Pressure")
        input_data["chol"] = cols[2].number_input("Cholesterol")
        input_data["thalach"] = cols[0].number_input("Max Heart Rate")
        input_data["oldpeak"] = cols[1].number_input("ST Depression")

        sex_map = {"Female": 0, "Male": 1}
        cp_map = {"Typical Angina":0, "Atypical Angina":1, "Non-Anginal Pain":2, "Asymptomatic":3}
        fbs_map = {"False":0, "True":1}
        restecg_map = {"Normal":0, "ST-T Abnormality":1, "Left Ventricular Hypertrophy":2}
        exang_map = {"No":0, "Yes":1}
        slope_map = {"Upsloping":0, "Flat":1, "Downsloping":2}
        thal_map = {"Normal":0, "Fixed Defect":1, "Reversible Defect":2, "Other":3}

        input_data["sex"] = sex_map[cols[2].selectbox("Sex", list(sex_map.keys()))]
        input_data["cp"] = cp_map[cols[0].selectbox("Chest Pain Type", list(cp_map.keys()))]
        input_data["fbs"] = fbs_map[cols[1].selectbox("Fasting Blood Sugar > 120 mg/dl?", list(fbs_map.keys()))]
        input_data["restecg"] = restecg_map[cols[2].selectbox("Rest ECG", list(restecg_map.keys()))]
        input_data["exang"] = exang_map[cols[0].selectbox("Exercise Induced Angina", list(exang_map.keys()))]
        input_data["slope"] = slope_map[cols[1].selectbox("Slope", list(slope_map.keys()))]
        input_data["ca"] = cols[2].number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3)
        input_data["thal"] = thal_map[cols[0].selectbox("Thalassemia", list(thal_map.keys()))]

    # -------------------------------
    # Kidney Disease Inputs
    # -------------------------------
    elif dataset == "kidney":
        cols = st.columns(3)
        input_data["age"] = cols[0].number_input("Age")
        input_data["bp"] = cols[1].number_input("Blood Pressure")
        input_data["bgr"] = cols[2].number_input("Blood Glucose Random")
        input_data["bu"] = cols[0].number_input("Blood Urea")
        input_data["sc"] = cols[1].number_input("Serum Creatinine")
        input_data["hemo"] = cols[2].number_input("Hemoglobin")
        input_data["pcv"] = cols[0].number_input("Packed Cell Volume")
        input_data["wc"] = cols[1].number_input("White Blood Cell Count")
        input_data["rc"] = cols[2].number_input("Red Blood Cell Count")

        # Categorical mappings
        rbc_map = {"Normal":0, "Abnormal":1}
        pc_map = {"Normal":0, "Abnormal":1}
        pcc_map = {"Present":1, "Not Present":0}
        ba_map = {"Present":1, "Not Present":0}
        htn_map = {"Yes":1, "No":0}
        dm_map = {"Yes":1, "No":0}
        cad_map = {"Yes":1, "No":0}
        appet_map = {"Good":0, "Poor":1}
        pe_map = {"Yes":1, "No":0}
        ane_map = {"Yes":1, "No":0}
        al_map = {"Normal":0, "Abnormal":1}
        su_map = {"Normal":0, "Abnormal":1}

        input_data["rbc"] = rbc_map[cols[2].selectbox("Red Blood Cells", list(rbc_map.keys()))]
        input_data["pc"] = pc_map[cols[0].selectbox("Pus Cell", list(pc_map.keys()))]
        input_data["pcc"] = pcc_map[cols[1].selectbox("Pus Cell Clumps", list(pcc_map.keys()))]
        input_data["ba"] = ba_map[cols[2].selectbox("Bacteria", list(ba_map.keys()))]
        input_data["htn"] = htn_map[cols[0].selectbox("Hypertension", list(htn_map.keys()))]
        input_data["dm"] = dm_map[cols[1].selectbox("Diabetes Mellitus", list(dm_map.keys()))]
        input_data["cad"] = cad_map[cols[2].selectbox("Coronary Artery Disease", list(cad_map.keys()))]
        input_data["appet"] = appet_map[cols[0].selectbox("Appetite", list(appet_map.keys()))]
        input_data["pe"] = pe_map[cols[1].selectbox("Pedal Edema", list(pe_map.keys()))]
        input_data["ane"] = ane_map[cols[2].selectbox("Anemia", list(ane_map.keys()))]
        input_data["al"] = al_map[cols[0].selectbox("Albumin", list(al_map.keys()))]
        input_data["su"] = su_map[cols[1].selectbox("Sugar", list(su_map.keys()))]

    # -------------------------------
    # Diabetes Inputs
    # -------------------------------
    elif dataset == "diabetes":
        cols = st.columns(3)
        input_data["Pregnancies"] = cols[0].number_input("Pregnancies")
        input_data["Glucose"] = cols[1].number_input("Glucose")
        input_data["BloodPressure"] = cols[2].number_input("Blood Pressure")
        input_data["SkinThickness"] = cols[0].number_input("Skin Thickness")
        input_data["Insulin"] = cols[1].number_input("Insulin")
        input_data["BMI"] = cols[2].number_input("BMI")
        input_data["DiabetesPedigreeFunction"] = cols[0].number_input("DPF")
        input_data["Age"] = cols[1].number_input("Age")

    submit = st.form_submit_button("🔍 Predict")
    st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# Prediction
# ===============================
if submit:
    try:
        model = load_model(dataset)
        preprocessor = load_preprocessor(dataset)

        input_df = pd.DataFrame([input_data])
        transformed = preprocessor.transform(input_df)

        # Prediction
        prediction = model.predict(transformed)[0]

        # Probability (if classifier supports predict_proba)
        try:
            probs = model.predict_proba(transformed)[0]
            confidence = max(probs)
            st.progress(int(confidence * 100))
        except:
            confidence = None

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if prediction == 1:
            st.markdown(f"<div class='danger'>⚠ Disease Detected</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='success'>✅ No Disease Detected</div>", unsafe_allow_html=True)

        if confidence:
            st.write(f"Prediction confidence: {confidence*100:.2f}%")

        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")
