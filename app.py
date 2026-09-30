import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
import requests
from streamlit_lottie import st_lottie

# ----------------------------------------
# 1. Page Configuration & Custom CSS
# ----------------------------------------
st.set_page_config(
    page_title="AI Accident Predictor",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Glassmorphism & Premium Dark/Light Aesthetic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Sleek Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        padding: 50px 30px;
        border-radius: 20px;
        color: #ffffff;
        text-align: center;
        margin-bottom: 40px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.2);
        position: relative;
        overflow: hidden;
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%; left: -50%; width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 60%);
        animation: rotate 20s linear infinite;
    }

    @keyframes rotate {
        100% { transform: rotate(360deg); }
    }

    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 10px;
        background: linear-gradient(to right, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        position: relative;
        z-index: 1;
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        color: #94a3b8;
        font-weight: 300;
        position: relative;
        z-index: 1;
    }

    /* Glassmorphism Cards for Input/Output */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.1);
    }
    
    /* Result Cards */
    .result-card {
        padding: 40px 20px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 15px 30px rgba(0,0,0,0.15);
        animation: popIn 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    .card-fatal { background: linear-gradient(135deg, #7f1d1d 0%, #dc2626 100%); color: white; }
    .card-serious { background: linear-gradient(135deg, #b45309 0%, #f59e0b 100%); color: white; }
    .card-slight { background: linear-gradient(135deg, #064e3b 0%, #10b981 100%); color: white; }
    
    @keyframes popIn {
        0% { opacity: 0; transform: scale(0.9); }
        100% { opacity: 1; transform: scale(1); }
    }

    /* Recommendations Box */
    .action-box {
        background-color: #f8fafc;
        border-left: 5px solid #3b82f6;
        padding: 20px;
        border-radius: 8px;
        color: #0f172a;
        margin-top: 20px;
        font-weight: 500;
    }

    /* Primary Button override */
    div.stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        color: white;
        border: none;
        padding: 18px;
        font-size: 1.3rem;
        font-weight: 700;
        border-radius: 12px;
        width: 100%;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    div.stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 25px rgba(168, 85, 247, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------
# 2. Helper Functions
# ----------------------------------------
@st.cache_data
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

@st.cache_resource
def load_model():
    model_path = 'models/best_model.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

# Load Animations
lottie_ai = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_UJNc2t.json")
lottie_car = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_41zblq.json")

# ----------------------------------------
# 3. Sidebar
# ----------------------------------------
with st.sidebar:
    if lottie_ai:
        st_lottie(lottie_ai, height=150, key="ai_anim")
    
    st.markdown("### 🛠️ System Architecture")
    st.markdown("---")
    st.markdown("""
    **Core Algorithm:** Logistic Regression  
    **Imbalance Handling:** SMOTE (Synthetic Minority Over-sampling)  
    **Evaluation Focus:** Macro Recall (Prioritizing Rare Events)
    """)
    st.markdown("---")
    st.success("Designed for City Transport Authorities to deploy targeted interventions at high-risk hotspots.")

# ----------------------------------------
# 4. Hero Section
# ----------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Nexus AI: Road Safety Intelligence</div>
    <div class="hero-subtitle">Predictive analytics for accident severity and infrastructural planning</div>
</div>
""", unsafe_allow_html=True)

model = load_model()
if model is None:
    st.error("🚨 Critical Error: Prediction Engine (best_model.pkl) is offline. Please run the training pipeline.")
    st.stop()

# ----------------------------------------
# 5. Main Application Tabs
# ----------------------------------------
tab_predict, tab_eda, tab_insights = st.tabs(["🎯 Live Prediction Engine", "📊 Exploratory Data Analysis", "🧠 Model Intelligence"])

with tab_predict:
    st.markdown("### 🌍 Environmental & Situational Telemetry")
    
    with st.container():
        # Using a sleek 3-column layout for inputs
        c1, c2, c3 = st.columns(3)
        with c1:
            speed = st.select_slider("⚡ Velocity Limit (mph)", options=[20, 30, 40, 50, 60, 70], value=30)
            weather = st.selectbox("☁️ Atmospheric Conditions", ['Normal', 'Raining', 'Snowing', 'Fog or mist', 'Other', 'Unknown'])
        with c2:
            light = st.selectbox("💡 Illumination Level", ['Daylight', 'Darkness - lights lit', 'Darkness - no lighting', 'Darkness - lighting unknown'])
            surface = st.selectbox("🛣️ Surface Traction", ['Dry', 'Wet or damp', 'Snow', 'Ice', 'Flood over road'])
        with c3:
            vehicle = st.selectbox("🚙 Primary Vehicle Class", ['Car', 'Motorcycle', 'Bus/Coach', 'Goods vehicle', 'Pedal cycle', 'Other'])
            time_of_day = st.selectbox("🕒 Temporal Window", ['Morning', 'Afternoon', 'Evening', 'Night'])

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Big Predict Button centered
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        analyze_btn = st.button("Initialize Threat Analysis")

    st.markdown("<br>", unsafe_allow_html=True)

    if analyze_btn:
        input_data = pd.DataFrame({
            'Weather_Conditions': [weather],
            'Light_Conditions': [light],
            'Road_Surface_Conditions': [surface],
            'Speed_Limit': [speed],
            'Vehicle_Type': [vehicle],
            'Time_of_Day': [time_of_day]
        })
        
        with st.spinner("Executing neural analysis on environmental vectors..."):
            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]
            classes = model.classes_
            
            st.markdown("---")
            st.markdown("### 📡 Diagnostic Report")
            
            res_col1, res_col2 = st.columns([1.2, 1])
            
            with res_col1:
                # Dynamic Action Recommendations
                if prediction == 'Fatal':
                    st.markdown("""
                    <div class="result-card card-fatal">
                        <h1 style='margin:0; font-size:3rem;'>CRITICAL RISK (FATAL)</h1>
                        <p style='font-size:1.2rem; opacity:0.9;'>Immediate infrastructural review required.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div class="action-box">
                        <strong>🛡️ Recommended Automated Actions:</strong><br>
                        - Deploy extreme speed reduction measures.<br>
                        - Mandate immediate street lighting installation if applicable.<br>
                        - Dispatch advance emergency medical services (EMS) to this zone.
                    </div>
                    """, unsafe_allow_html=True)
                    
                elif prediction == 'Serious':
                    st.markdown("""
                    <div class="result-card card-serious">
                        <h1 style='margin:0; font-size:3rem;'>SERIOUS INJURY RISK</h1>
                        <p style='font-size:1.2rem; opacity:0.9;'>High probability of hospitalization.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div class="action-box">
                        <strong>🛡️ Recommended Automated Actions:</strong><br>
                        - Trigger variable message signs (VMS) warning drivers.<br>
                        - Increase automated traffic enforcement (cameras).
                    </div>
                    """, unsafe_allow_html=True)
                    
                else:
                    st.markdown("""
                    <div class="result-card card-slight">
                        <h1 style='margin:0; font-size:3rem;'>SLIGHT RISK</h1>
                        <p style='font-size:1.2rem; opacity:0.9;'>Accident probable, low severity expected.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div class="action-box">
                        <strong>🛡️ Recommended Automated Actions:</strong><br>
                        - Standard traffic policing and monitoring.<br>
                        - Routine road maintenance.
                    </div>
                    """, unsafe_allow_html=True)

            with res_col2:
                # Premium Donut Chart
                prob_df = pd.DataFrame({'Severity': classes, 'Probability': probabilities})
                color_map = {'Fatal': '#ef4444', 'Serious': '#f59e0b', 'Slight': '#10b981'}
                
                fig = px.pie(prob_df, values='Probability', names='Severity', 
                             hole=0.7, color='Severity', color_discrete_map=color_map)
                
                fig.update_traces(textposition='outside', textinfo='percent+label', 
                                  marker=dict(line=dict(color='white', width=2)),
                                  textfont_size=14)
                
                fig.update_layout(
                    showlegend=False, 
                    margin=dict(t=20, b=20, l=20, r=20),
                    height=300,
                    annotations=[dict(text='AI<br>Confidence', x=0.5, y=0.5, font_size=20, showarrow=False, font=dict(weight='bold'))]
                )
                
                st.plotly_chart(fig, use_container_width=True)

with tab_eda:
    st.markdown("### 📊 Dataset Exploration")
    st.write("Visualizing the internal patterns of the synthetic dataset.")
    
    # Load dataset for EDA (Assuming it's generated)
    data_path = 'data/road_accidents.csv'
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        
        c_eda1, c_eda2 = st.columns(2)
        with c_eda1:
            st.markdown("#### Severity Distribution (Original Imbalance)")
            sev_counts = df['Accident_Severity'].value_counts().reset_index()
            sev_counts.columns = ['Severity', 'Count']
            fig1 = px.bar(sev_counts, x='Severity', y='Count', color='Severity',
                          color_discrete_map={'Slight': '#10b981', 'Serious': '#f59e0b', 'Fatal': '#ef4444'})
            st.plotly_chart(fig1, use_container_width=True)
            
        with c_eda2:
            st.markdown("#### Fatalities by Speed Limit")
            fatal_df = df[df['Accident_Severity'] == 'Fatal']
            speed_counts = fatal_df['Speed_Limit'].value_counts().reset_index()
            speed_counts.columns = ['Speed', 'Count']
            fig2 = px.line(speed_counts.sort_values('Speed'), x='Speed', y='Count', markers=True,
                           line_shape='spline', line_dash_sequence=['solid'])
            fig2.update_traces(line_color='#ef4444', marker=dict(size=10))
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Dataset not found in `data/`. Run `generate_dataset.py` to view live EDA.")

with tab_insights:
    st.markdown("### 🧠 The Intelligence Behind Nexus AI")
    
    st.markdown("""
    #### 1. The Imbalance Problem
    In real-world road networks, slight accidents heavily outnumber fatal ones (typically 80% to 5%). 
    If a naive AI predicts "Slight" every time, it achieves 80% accuracy, but completely fails its safety objective.
    
    #### 2. The SMOTE Solution
    To force the AI to learn the specific environmental triggers of a Fatal accident, we used **Synthetic Minority Over-sampling Technique (SMOTE)**. 
    This algorithm mathematically synthesizes new 'Fatal' data points in the multi-dimensional feature space, balancing the scales so the AI treats a fatality with the mathematical respect it deserves.
    
    #### 3. Why Logistic Regression?
    In our cross-validation trials across 5 distinct algorithms (including Decision Trees and Ensembles like Random Forest), Logistic Regression—when paired with SMOTE—yielded the **highest Macro Recall**. 
    
    *Translation for stakeholders: It minimizes False Negatives. It is the least likely to look at a highly dangerous intersection and accidentally declare it "Safe".*
    """)
