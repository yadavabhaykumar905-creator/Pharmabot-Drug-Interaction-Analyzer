import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import urllib.request


model_path = 'ddi_model.pkl'

if not os.path.exists(model_path):
    with st.spinner("Downloading model from external link... Please wait."):
        
        url = "https://huggingface.co/abhay109/ddi_model/resolve/main/ddi_model.pkl"
        urllib.request.urlretrieve(url, model_path)


model = joblib.load(model_path)



le_drug1 = joblib.load('le_drug1.pkl')
le_drug2 = joblib.load('le_drug2.pkl')
le_target = joblib.load('le_target.pkl')

st.set_page_config(
    page_title="Pharmabot AI: Drug Interaction Analyzer",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #007bff;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .reaction-card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 12px;
        border-left: 6px solid #dc3545;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        word-wrap: break-word;
        white-space: normal;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("💊 Pharmabot AI: Intelligent Drug Interaction & Contraindication Analyzer")
st.markdown("Advanced clinical pharmacovigilance engine designed to detect multi-drug adverse interactions and severity risks.")

with st.sidebar:
    st.image("https://img.icons8.com/color/96/medical-doctor.png", width=80)
    st.header("Clinical Profile")
    st.markdown("---")
    patient_age = st.number_input("Patient Age", min_value=1, max_value=120, value=45)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    condition = st.selectbox("Underlying Medical Condition", ["Hypertension", "Diabetes", "Heart Failure", "Asthma", "None"])
    st.markdown("---")
    st.info("Select active medications from the dropdown to compute precise interaction metrics.")

available_drugs = list(le_drug1.classes_)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🧬 Primary Medication")
    drug_1 = st.selectbox("Select Drug 1", available_drugs, index=0)

with col2:
    st.markdown("### 🧬 Secondary Medication")
    drug_2 = st.selectbox("Select Drug 2", available_drugs, index=1 if len(available_drugs) > 1 else 0)

st.markdown("<br>", unsafe_allow_html=True)

if st.button("Run Clinical Interaction Analysis"):
    with st.spinner("Analyzing pharmacological data and computing risk probability..."):
        
        d1_val = le_drug1.transform([drug_1])[0] if drug_1 in le_drug1.classes_ else 0
        d2_val = le_drug2.transform([drug_2])[0] if drug_2 in le_drug2.classes_ else 0
        
        input_data = pd.DataFrame([[d1_val, d2_val]], columns=['drug_1_enc', 'drug_2_enc'])
        
        prediction_idx = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)
        
        max_prob = float(np.max(probability) * 100)
        predicted_reaction = le_target.inverse_transform([prediction_idx])[0]
        
        st.markdown("---")
        st.subheader("📊 Analysis Results & Scorecard")
        
        st.markdown(
            f"""
            <div class="reaction-card">
                <h3 style="margin-top:0; color:#dc3545;">⚠️ Adverse Clinical Interaction Identified</h3>
                <p style="font-size:18px; font-weight:bold; color:#222; margin-bottom: 8px;">
                    Predicted Interaction: <span style="color:#0056b3; word-break: break-all;">{predicted_reaction}</span>
                </p>
                <p style="font-size:15px; color:#444; margin-bottom: 15px;">
                    Confidence Score: <b>{max_prob:.1f}%</b>
                </p>
                <hr style="border: none; border-top: 1px solid #eee; margin:12px 0;">
                <p style="color:#333; line-height: 1.5; font-size: 15px;">
                    The combination of <b>{drug_1}</b> and <b>{drug_2}</b> under condition <b>{condition}</b> has been analyzed. The specific interaction description predicted by the model is: <b>{predicted_reaction}</b>.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
                
        st.markdown("---")
        st.caption("Pharmabot AI Pharmacovigilance Suite | Built for Clinical Decision Support")
