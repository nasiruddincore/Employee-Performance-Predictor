import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px

st.set_page_config(page_title="HR Analytics App", layout="wide")
st.title("📈 Employee Performance Predictor")
st.markdown("Use this tool to forecast employee performance bands and assign L&D interventions.")

MODEL_PATH = "models/employee_perf_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("⚠️ Model not found. Run `python src/train_model.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)

st.sidebar.header("Employee Metrics Input")
department = st.sidebar.selectbox("Department", ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance'])
job_level = st.sidebar.selectbox("Job Level", ['Junior', 'Mid', 'Senior'])
experience = st.sidebar.slider("Experience (Years)", 0, 45, 5)
training = st.sidebar.slider("Training Hours (Last 6 Months)", 0, 100, 20)
delivery = st.sidebar.slider("On-Time Delivery Rate (0-1)", 0.0, 1.0, 0.8)
manager_score = st.sidebar.slider("Previous Manager Score (1-5)", 1.0, 5.0, 3.5)
bugs = st.sidebar.slider("Errors / Bugs Count", 0, 50, 5)

if st.sidebar.button("Predict Performance"):
    input_data = pd.DataFrame({
        'department': [department],
        'job_level': [job_level],
        'experience_years': [experience],
        'training_hours': [training],
        'on_time_delivery_rate': [delivery],
        'manager_score': [manager_score],
        'bugs_or_errors': [bugs]
    })
    
    # Get both the final prediction and the exact probability scores
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    classes = model.classes_
    
    st.markdown("---")
    st.subheader("🎯 Prediction Results & Analytics")
    
    # 1. Main Status Banner
    if prediction == 'High':
        st.success(f"🌟 Predicted Band: **{prediction}** - Candidate for Fast-Track Promotion.")
    elif prediction == 'Medium':
        st.info(f"✅ Predicted Band: **{prediction}** - Steady performer. Maintain current trajectory.")
    else:
        st.error(f"⚠️ Predicted Band: **{prediction}** - Action Required.")
        
    # 2. Visual Layout using Columns
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown("### 📊 AI Confidence Distribution")
        # Create a DataFrame for the interactive Plotly chart
        prob_df = pd.DataFrame({
            'Performance Band': classes,
            'Probability (%)': probabilities * 100
        })
        
        # Plotly Bar Chart mapping colors to performance bands
        fig = px.bar(
            prob_df, 
            x='Performance Band', 
            y='Probability (%)', 
            color='Performance Band',
            color_discrete_map={'High': '#28a745', 'Medium': '#17a2b8', 'Low': '#dc3545'},
            text_auto='.1f'
        )
        fig.update_layout(showlegend=False, height=350, margin=dict(t=20, b=0, l=0, r=0))
        fig.update_traces(textfont_size=14, textangle=0, textposition="outside", cliponaxis=False)
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.markdown("### 🛠️ Recommended HR Actions")
        if prediction == 'High':
            st.write("✔️ **Promotion:** Add to leadership fast-track pipeline.")
            st.write("✔️ **Compensation:** Review for cycle bonus eligibility.")
            st.write("✔️ **Engagement:** Assign as a mentor to junior staff.")
        elif prediction == 'Medium':
            st.write("✔️ **Development:** Offer optional upskilling certifications.")
            st.write("✔️ **Review:** Continue standard quarterly evaluations.")
        else:
            st.write("✔️ **Intervention:** Schedule an immediate 1-on-1 manager coaching session.")
            st.write("✔️ **Training:** Assign targeted time-management and quality-assurance workshops.")
            st.write("✔️ **Tracking:** Place on a 30-day supportive Performance Improvement Plan (PIP).")