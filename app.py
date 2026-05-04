import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from fpdf import FPDF
import os
import datetime
import uuid
import logging

try:
    from google import genai
    from google.genai import types
except ImportError:
    st.error("Please run `pip install google-genai` to use the generative features.")

# Load model files - Core Variables Maintained
model = joblib.load("KNN_heart_model.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

st.set_page_config(page_title="Heart Predictor Pro", page_icon="❤️", layout="wide")

# --- 1. PREMIUM CSS ---
st.markdown("""
<style>
    /* Dark Theme with Red Accents & Glassmorphism */
    .stApp {
        background-color: #0b0e14;
        color: #e0e0e0;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ff3b3b !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Cards and Containers */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease;
        margin-bottom: 1em;
        height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: #ff3b3b;
    }
    .metric-title {
        font-size: 1.1rem;
        color: #a0a0a0;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #ffffff;
    }
    
    /* Primary Action Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #ff3b3b 0%, #d32f2f 100%);
        color: white;
        border-radius: 12px;
        height: 3.5em;
        width: 100%;
        font-size: 18px;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 15px rgba(255, 59, 59, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 59, 59, 0.6);
        color: #ffffff;
        border: none;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #121822;
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .fade-in {
        animation: fadeIn 0.8s ease forwards;
    }
    
    /* Custom divider */
    hr {
        border-color: rgba(255,59,59,0.2) !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. HELPER FUNCTIONS ---
def calculate_health_score(age, bp, chol, max_hr):
    score = 100
    # Continuous deduction for hyper-sensitivity so it changes on every input
    score -= abs(bp - 120) * 0.4
    score -= abs(chol - 160) * 0.15
    if age > 25: score -= (age - 25) * 0.3
    target_hr = 220 - age
    score -= abs(max_hr - (target_hr * 0.8)) * 0.2
    return max(10, min(100, int(score)))

def generate_rule_based_advice(risk_level, bp, chol):
    advice = []
    if risk_level == "High":
        advice.append("- Immediate Consultation: Please consult a cardiologist as soon as possible.")
        advice.append("- Strict Diet: Follow a low-sodium, heart-healthy diet strictly.")
        advice.append("- Activity: Avoid strenuous activity until physician clearance.")
    elif risk_level == "Medium":
        advice.append("- Regular Checkups: Schedule a routine checkup soon.")
        advice.append("- Diet Improvement: Reduce intake of saturated fats and salt.")
        advice.append("- Exercise: Engage in moderate daily activity like walking.")
    else:
        advice.append("- Maintain Routine: Continue your current healthy lifestyle.")
        advice.append("- Prevention: Maintain an active lifestyle and balanced diet.")
    
    if bp > 130:
        advice.append("- Blood Pressure Warning: Your BP is slightly elevated. Monitor regularly and reduce sodium.")
    if chol > 240:
        advice.append("- Cholesterol Warning: Cholesterol levels need attention. Avoid trans fats and eat more fiber.")
    
    return "\n".join(advice)

def generate_pdf_report(patient_details, prob, risk_level, health_score, ai_advice=None):
    pdf = FPDF()
    pdf.add_page()
    
    # Colors and Fonts
    pdf.set_fill_color(240, 240, 240)
    pdf.set_text_color(20, 20, 20)
    
    # Header
    pdf.set_font("Arial", 'B', 22)
    pdf.cell(0, 15, txt="COMPREHENSIVE CARDIAC HEALTH REPORT", ln=True, align='C')
    pdf.set_font("Arial", 'I', 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, txt="Advanced Cardiovascular Risk Assessment System", ln=True, align='C')
    pdf.ln(5)
    
    # Meta info
    report_id = str(uuid.uuid4())[:8].upper()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 8, txt=f"Date: {timestamp} | Report ID: #{report_id}", ln=True, align='R')
    pdf.ln(5)
    
    # Section: Patient Details
    pdf.set_text_color(255, 59, 59)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, txt="1. PATIENT DETAILS", ln=True)
    pdf.set_text_color(20, 20, 20)
    pdf.set_font("Arial", '', 11)
    
    details_str = (f"Age: {patient_details['Age']}      "
                   f"Sex: {patient_details['Sex']}      "
                   f"Resting BP: {patient_details['RestingBP']} mmHg      "
                   f"Cholesterol: {patient_details['Cholesterol']} mg/dl")
    pdf.cell(0, 8, txt=details_str, ln=True)
    details_str_2 = (f"Max HR: {patient_details['MaxHR']}      "
                     f"Chest Pain: {patient_details['ChestPainType']}      "
                     f"Exercise Angina: {patient_details['ExerciseAngina']}")
    pdf.cell(0, 8, txt=details_str_2, ln=True)
    pdf.ln(5)
    
    # Section: Risk Analysis
    pdf.set_text_color(255, 59, 59)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, txt="2. CARDIAC RISK ANALYSIS", ln=True)
    pdf.set_text_color(20, 20, 20)
    pdf.set_font("Arial", '', 12)
    
    pdf.cell(0, 8, txt=f"Predicted Risk Level: {risk_level} Risk", ln=True)
    pdf.cell(0, 8, txt=f"Probability Score: {round(prob*100, 2)}%", ln=True)
    pdf.cell(0, 8, txt=f"Overall Health Score: {health_score} / 100", ln=True)
    pdf.ln(5)
    
    # Ensure proper spacing for next section
    pdf.ln(5)

    # Section: Recommendations
    pdf.set_text_color(255, 59, 59)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, txt="3. HEALTH RECOMMENDATIONS & LIFESTYLE", ln=True)
    pdf.set_text_color(20, 20, 20)
    pdf.set_font("Arial", '', 11)
    
    # Rule based advice
    rule_advice = generate_rule_based_advice(risk_level, patient_details['RestingBP'], patient_details['Cholesterol'])
    pdf.multi_cell(0, 8, txt="Standard Guidelines:\n" + rule_advice)
    pdf.ln(5)
    
    # Clinical Recommendations
    if ai_advice:
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(0, 8, txt="Clinical Insights & Recommendations:", ln=True)
        pdf.set_font("Arial", '', 11)
        pdf.multi_cell(0, 6, txt=ai_advice.encode('latin-1', 'replace').decode('latin-1'))
    
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 9)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 5, txt="Disclaimer: This is an automated assessment report and should not replace professional medical diagnosis.", ln=True, align='C')

    file_path = "report.pdf"
    pdf.output(file_path)
    return file_path


# --- 3. LAYOUT & SIDEBAR ---
col_head1, col_head2 = st.columns([1, 10])
with col_head1:
    st.markdown("<h1 style='text-align:center;'>❤️</h1>", unsafe_allow_html=True)
with col_head2:
    st.title("Advanced Cardiology Dashboard")
    st.markdown("<p style='color:#a0a0a0; font-size:1.1rem;'>Professional Data-Driven Cardiac Risk Analysis</p>", unsafe_allow_html=True)

st.sidebar.header("📋 Patient Profile")

# Gemini API Key
gemini_api_key = st.secrets.get("gemini_api_key")

# Input Fields
col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    # Adding slider options converted to number inputs for better mobile/premium feel. 
    # Let's keep original names but adjust ranges softly to match initial code.
    age = st.number_input("Age", 18, 100, 40)
    resting_bp = st.number_input("Resting BP", 80, 200, 120)
    fasting_bs = st.selectbox("Fasting BS > 120", [0, 1])
    oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0, step=0.1)
with col_sb2:
    sex = st.selectbox("Sex", ['M', 'F'])
    cholestrol = st.number_input("Cholesterol", 100, 600, 200)
    max_hr = st.number_input("Max HR", 60, 220, 150)

st.sidebar.markdown("---")    
chest_pain = st.sidebar.selectbox("Chest Pain", ["ATA", "NAP", "ASY", "TA"])
resting_ecg = st.sidebar.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
exercise_angina = st.sidebar.selectbox("Exercise Angina", ["Y", "N"])
st_slope = st.sidebar.selectbox("ST Slope", ["Up", "Flat", "Down"])

# --- 4. TOP METRICS DASHBOARD ---
st.markdown("### Patient Metrics Summary")
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f"<div class='metric-card fade-in'><div class='metric-title'>Age</div><div class='metric-value'>{age}</div></div>", unsafe_allow_html=True)
with m2:
    bp_color = '#ff3b3b' if resting_bp > 130 else '#4caf50'
    bp_text = 'Elevated' if resting_bp > 130 else 'Optimal'
    st.markdown(f"<div class='metric-card fade-in' style='animation-delay: 0.1s;'><div class='metric-title'>Resting BP</div><div class='metric-value'>{resting_bp}</div><div style='color:{bp_color}; font-size:0.8rem; margin-top:5px;'>{bp_text}</div></div>", unsafe_allow_html=True)
with m3:
    chol_color = '#ff3b3b' if cholestrol > 240 else '#4caf50'
    chol_text = 'High' if cholestrol > 240 else 'Healthy'
    st.markdown(f"<div class='metric-card fade-in' style='animation-delay: 0.2s;'><div class='metric-title'>Cholesterol</div><div class='metric-value'>{cholestrol}</div><div style='color:{chol_color}; font-size:0.8rem; margin-top:5px;'>{chol_text}</div></div>", unsafe_allow_html=True)
with m4:
    st.markdown(f"<div class='metric-card fade-in' style='animation-delay: 0.3s;'><div class='metric-title'>Max HR</div><div class='metric-value'>{max_hr}</div></div>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# --- 5. PREDICTION LOGIC ---
raw_input = {
    'Age': age,
    'RestingBP': resting_bp,
    'Cholesterol': cholestrol,
    'FastingBS': fasting_bs,
    'MaxHR': max_hr,
    'Oldpeak': oldpeak,
    'Sex_' + sex: 1,
    'ChestPainType_' + chest_pain: 1,
    'RestingECG_' + resting_ecg: 1,
    'ExerciseAngina_' + exercise_angina: 1,
    'ST_Slope_' + st_slope: 1
}

# Preserve the exact preprocessing structure per user requirements
input_df = pd.DataFrame([raw_input])
for col in expected_columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[expected_columns]
scaled_input = scaler.transform(input_df)

if st.button("🚀 Analyze Patient Risk"):
    with st.spinner("🤖 Processing health parameters using AI Models..."):
        # Inference
        prediction = model.predict(scaled_input)[0]
        prob = model.predict_proba(scaled_input)[0][1]
        
        # Risk assessment thresholds
        if prob < 0.3:
            risk_level = "Low"
            color = "#4caf50"
        elif prob < 0.7:
            risk_level = "Medium"
            color = "#ff9800"
        else:
            risk_level = "High"
            color = "#f44336"
            
        health_score = calculate_health_score(age, resting_bp, cholestrol, max_hr)
        
        # Clinical Recommendations Generation
        ai_advice = None
        if gemini_api_key:
            try:
                client = genai.Client(api_key=gemini_api_key)
                prompt = (f"Act as a professional cardiologist. A {age} year old {sex} has a {risk_level} risk "
                          f"({round(prob*100,1)}%) of heart disease. Resting BP is {resting_bp}, Cholesterol is {cholestrol}. "
                          f"Provide 3 brief, actionable health and lifestyle recommendations. Be professional, direct, and empathetic.")
                completion = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.7)
                )
                ai_advice = completion.text
            except Exception as e:
                st.error(f"Inference Engine Error: {str(e)}")
                ai_advice = "Recommendation service unavailable. Please rely on the standard rules above."
        
        # --- 6. VISUALIZATIONS ---
        col_res1, col_res2 = st.columns([1, 1])
        
        with col_res1:
            st.markdown("<h3 class='fade-in'>🎯 Risk Assessment</h3>", unsafe_allow_html=True)
            # Plotly Gauge Chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"{risk_level} Risk", 'font': {'size': 24, 'color': color}},
                number = {'suffix': "%", 'font': {'color': color, 'size': 40}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white", 'tickfont': {'color': 'white'}},
                    'bar': {'color': "rgba(255,255,255,0.7)", 'thickness': 0.25},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 2,
                    'bordercolor': "#333",
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(76, 175, 80, 0.4)'},     # Green
                        {'range': [30, 70], 'color': 'rgba(255, 152, 0, 0.4)'},    # Yellow
                        {'range': [70, 100], 'color': 'rgba(244, 67, 54, 0.4)'}],  # Red
                }
            ))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=320, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)

        with col_res2:
            st.markdown("<h3 class='fade-in'>🩺 Diagnostic Insights</h3>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-card fade-in' style='border-left: 5px solid {color}; animation-delay: 0.2s;'>"
                        f"<div class='metric-title'>Overall Health Score</div>"
                        f"<div class='metric-value' style='color:{color};'>{health_score} <span style='font-size:1rem;color:#ccc;'>/ 100</span></div>"
                        f"</div>", unsafe_allow_html=True)
            
            # Clinical Summary Alert
            if risk_level == "High":
                st.error("🚨 **Critical Attention:** High probability of a cardiovascular event. Follow up with a specialist immediately.")
            elif risk_level == "Medium":
                st.warning("⚠️ **Elevated Risk:** Lifestyle changes and clinical monitoring are highly recommended.")
            else:
                st.success("✅ **Low Risk:** Heart vitals are within safe thresholds. Keep up the healthy habits.")
                
            if ai_advice:
                st.info(f"🩺 **Clinical Recommendations:**\n\n{ai_advice}")
        
        # PDF Generation
        patient_details = {
            'Age': age, 'Sex': sex, 'RestingBP': resting_bp, 'Cholesterol': cholestrol,
            'MaxHR': max_hr, 'ChestPainType': chest_pain, 'ExerciseAngina': exercise_angina
        }
        report_path = generate_pdf_report(patient_details, prob, risk_level, health_score, ai_advice)
        
        st.markdown("<br>", unsafe_allow_html=True)
        with open(report_path, "rb") as f:
            st.download_button("📄 Download Comprehensive Medical Report (PDF)", f, file_name="Health_Diagnostic_Report.pdf", mime="application/pdf")

# --- 7. HEALTH CHAT ASSISTANT ---
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader("💬 Health Care Assistant")
st.markdown("<p style='color:#a0a0a0; font-size:0.9rem;'>Ask questions about your health, diet, or cardiology guidelines.</p>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Type your health question here..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if gemini_api_key:
            try:
                client = genai.Client(api_key=gemini_api_key)
                # Convert Streamlit history to Gemini format
                contents = []
                for m in st.session_state.messages:
                    role = "user" if m["role"] == "user" else "model"
                    contents.append(types.Content(role=role, parts=[types.Part.from_text(text=m["content"])]))
                
                with st.spinner("Typing..."):
                    completion = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=contents,
                        config=types.GenerateContentConfig(
                            system_instruction="You are a professional healthcare assistant specializing in cardiology. Be concise, friendly, and structure answers well.",
                            temperature=0.6,
                        )
                    )
                    response = completion.text
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Service communication error: {str(e)}")
        else:
            response = "⚠️ The generative service is unconfigured. For now, remember that heart health heavily relies on exercise, a balanced diet, and stress management."
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown("<br><br><p style='text-align:center;color:#555;font-size:0.8rem;'>⚠️ Disclaimer: This application is for informational purposes only and does NOT constitute professional medical advice.</p>", unsafe_allow_html=True)

            
