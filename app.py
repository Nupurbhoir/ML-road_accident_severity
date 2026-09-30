import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------------------
# 1. Page Configuration & Ultra-Premium CSS
# ----------------------------------------
st.set_page_config(
    page_title="Accident Severity Predictor",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS matching exact dark theme from UI screenshots
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Sleek Hero Banner */
    .hero-box {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 45px 35px;
        border-radius: 20px;
        color: #ffffff;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.4);
    }

    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        margin-bottom: 12px;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }
    
    .hero-subtitle {
        font-size: 1.25rem;
        color: #94a3b8;
        font-weight: 300;
    }

    /* Selected Model Callout Badge */
    .model-callout {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.12) 0%, rgba(168, 85, 247, 0.12) 100%);
        border: 1px solid rgba(129, 140, 248, 0.35);
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 25px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
    }

    .model-callout-title {
        color: #818cf8;
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .model-callout-text {
        color: #cbd5e1;
        font-size: 0.95rem;
        margin: 0;
    }

    /* Input Telemetry Card */
    .telemetry-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 25px;
        margin-bottom: 25px;
    }

    /* Result Cards */
    .result-card {
        padding: 35px 20px;
        border-radius: 18px;
        text-align: center;
        box-shadow: 0 12px 28px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }
    
    .card-fatal { background: linear-gradient(135deg, #991b1b 0%, #ef4444 100%); color: white; }
    .card-serious { background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%); color: white; }
    .card-slight { background: linear-gradient(135deg, #059669 0%, #10b981 100%); color: white; }
    
    .action-box {
        background: rgba(255, 255, 255, 0.04);
        border-left: 4px solid #38bdf8;
        padding: 18px 20px;
        border-radius: 10px;
        color: #e2e8f0;
        margin-top: 15px;
        font-size: 0.98rem;
        line-height: 1.6;
    }

    /* Sidebar Styling */
    .sidebar-card {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 16px;
        border-radius: 12px;
        color: #6ee7b7;
        font-size: 0.92rem;
        line-height: 1.5;
        margin-top: 20px;
    }

    /* Primary Action Button */
    div.stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        padding: 16px 28px;
        font-size: 1.15rem;
        font-weight: 700;
        border-radius: 12px;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.3);
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 28px rgba(99, 102, 241, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------
# 2. Model Loading
# ----------------------------------------
@st.cache_resource
def load_trained_model():
    model_path = 'models/best_model.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_trained_model()

# ----------------------------------------
# 3. Sidebar Configuration
# ----------------------------------------
with st.sidebar:
    st.markdown("### 🛠️ System Architecture")
    st.markdown("---")
    st.markdown("""
    **Core Algorithm:** Logistic Regression  
    **Imbalance Handling:** SMOTE  
    *(Synthetic Minority Over-sampling)*  
    **Evaluation Focus:** Macro Recall  
    *(Prioritizing Rare Events)*
    """)
    st.markdown("---")
    st.markdown("""
    <div class="sidebar-card">
        <b>Designed for City Transport Authorities</b> to deploy targeted interventions at high-risk hotspots.
    </div>
    """, unsafe_allow_html=True)

# ----------------------------------------
# 4. Hero Banner
# ----------------------------------------
st.markdown("""
<div class="hero-box">
    <div class="hero-title">Nexus AI: Road Safety Intelligence</div>
    <div class="hero-subtitle">Predictive analytics for accident severity and infrastructural planning</div>
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs matching exact uploaded screenshots
tab_predict, tab_eda, tab_insights = st.tabs([
    "🎯 Live Prediction Engine", 
    "📊 Exploratory Data Analysis", 
    "🧠 Model Intelligence"
])

# ----------------------------------------
# TAB 1: LIVE PREDICTION ENGINE
# ----------------------------------------
with tab_predict:
    st.markdown("### 🌍 Environmental & Situational Telemetry")
    
    with st.container():
        c1, c2, c3 = st.columns(3)
        with c1:
            speed = st.slider("⚡ Velocity Limit (mph)", min_value=20, max_value=70, value=30, step=10)
            weather = st.selectbox("☁️ Atmospheric Conditions", ['Normal', 'Raining', 'Snowing', 'Fog or mist', 'Other', 'Unknown'])
        with c2:
            light = st.selectbox("💡 Illumination Level", ['Daylight', 'Darkness - lights lit', 'Darkness - no lighting', 'Darkness - lighting unknown'])
            surface = st.selectbox("🛣️ Surface Traction", ['Dry', 'Wet or damp', 'Snow', 'Ice', 'Flood over road'])
        with c3:
            vehicle = st.selectbox("🚙 Primary Vehicle Class", ['Car', 'Motorcycle', 'Bus/Coach', 'Goods vehicle', 'Pedal cycle', 'Other'])
            time_of_day = st.selectbox("🕒 Temporal Window", ['Morning', 'Afternoon', 'Evening', 'Night'])

    st.markdown("<br>", unsafe_allow_html=True)
    
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        analyze_btn = st.button("INITIALIZE THREAT ANALYSIS")

    st.markdown("<br>", unsafe_allow_html=True)

    if analyze_btn:
        if model is None:
            st.error("Model binary not found! Please ensure `models/best_model.pkl` exists.")
        else:
            input_data = pd.DataFrame({
                'Weather_Conditions': [weather],
                'Light_Conditions': [light],
                'Road_Surface_Conditions': [surface],
                'Speed_Limit': [speed],
                'Vehicle_Type': [vehicle],
                'Time_of_Day': [time_of_day]
            })
            
            with st.spinner("Processing vector telemetry through Logistic Regression + SMOTE pipeline..."):
                prediction = model.predict(input_data)[0]
                probabilities = model.predict_proba(input_data)[0]
                classes = model.classes_
                
                st.markdown("---")
                
                # Prominently Display Selected Model
                st.markdown("""
                <div class="model-callout">
                    <div class="model-callout-title">🤖 Selected Model: Logistic Regression + SMOTE</div>
                    <div class="model-callout-text">
                        <b>Validated Benchmark Metrics:</b> Fatal Recall: 78.80% | Serious Recall: 60.44% | Macro F1-Score: 64.72% | Overall CV Accuracy: 77.23%
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### 📡 Diagnostic Report")
                res_col1, res_col2 = st.columns([1.2, 1])
                
                with res_col1:
                    if prediction == 'Fatal':
                        st.markdown("""
                        <div class="result-card card-fatal">
                            <h1 style='margin:0; font-size:3.2rem; font-weight:800;'>FATAL RISK</h1>
                            <p style='font-size:1.2rem; opacity:0.9; margin-top:8px;'>Critical severity expected. Immediate emergency dispatch required.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("""
                        <div class="action-box">
                            <strong>🛡️ Recommended Automated Actions:</strong><br>
                            • Dispatch Level-1 Emergency Trauma Units.<br>
                            • Trigger automated speed calming and variable lighting.<br>
                            • Initiate immediate infrastructural safety audit.
                        </div>
                        """, unsafe_allow_html=True)
                        
                    elif prediction == 'Serious':
                        st.markdown("""
                        <div class="result-card card-serious">
                            <h1 style='margin:0; font-size:3.2rem; font-weight:800;'>SERIOUS RISK</h1>
                            <p style='font-size:1.2rem; opacity:0.9; margin-top:8px;'>High probability of hospitalization.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("""
                        <div class="action-box">
                            <strong>🛡️ Recommended Automated Actions:</strong><br>
                            • Dispatch standard EMS units.<br>
                            • Activate Variable Message Signs (VMS) warning drivers.<br>
                            • Increase automated enforcement presence.
                        </div>
                        """, unsafe_allow_html=True)
                        
                    else:
                        st.markdown("""
                        <div class="result-card card-slight">
                            <h1 style='margin:0; font-size:3.2rem; font-weight:800;'>SLIGHT RISK</h1>
                            <p style='font-size:1.2rem; opacity:0.9; margin-top:8px;'>Accident probable, low severity expected.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("""
                        <div class="action-box">
                            <strong>🛡️ Recommended Automated Actions:</strong><br>
                            • Standard traffic policing and monitoring.<br>
                            • Routine road surface maintenance log update.
                        </div>
                        """, unsafe_allow_html=True)

                with res_col2:
                    prob_df = pd.DataFrame({'Severity': classes, 'Probability': probabilities})
                    color_map = {'Fatal': '#ef4444', 'Serious': '#f59e0b', 'Slight': '#10b981'}
                    
                    fig = px.pie(prob_df, values='Probability', names='Severity', 
                                 hole=0.68, color='Severity', color_discrete_map=color_map)
                    
                    fig.update_traces(textposition='outside', textinfo='percent+label', 
                                      marker=dict(line=dict(color='#0f172a', width=3)),
                                      textfont_size=13)
                    
                    fig.update_layout(
                        showlegend=False, 
                        margin=dict(t=20, b=20, l=20, r=20),
                        height=290,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        annotations=[dict(text='AI<br>Confidence', x=0.5, y=0.5, font_size=18, showarrow=False, font=dict(weight='bold', color='#ffffff'))]
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------
# TAB 2: EXPLORATORY DATA ANALYSIS
# ----------------------------------------
with tab_eda:
    st.markdown("### 📊 Dataset Exploration")
    st.write("Visualizing the internal patterns of the synthetic dataset.")
    
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
            fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig1, use_container_width=True)
            
        with c_eda2:
            st.markdown("#### Fatalities by Speed Limit")
            fatal_df = df[df['Accident_Severity'] == 'Fatal']
            speed_counts = fatal_df['Speed_Limit'].value_counts().reset_index()
            speed_counts.columns = ['Speed', 'Count']
            fig2 = px.line(speed_counts.sort_values('Speed'), x='Speed', y='Count', markers=True,
                           line_shape='spline')
            fig2.update_traces(line_color='#ef4444', marker=dict(size=10))
            fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Additional Environmental Risk Visualizations")
        
        c_eda3, c_eda4 = st.columns(2)
        with c_eda3:
            if os.path.exists('assets/severity_vs_light.png'):
                st.image('assets/severity_vs_light.png', caption="Severity vs Illumination Level", use_container_width=True)
        with c_eda4:
            if os.path.exists('assets/confusion_matrix.png'):
                st.image('assets/confusion_matrix.png', caption="Logistic Regression + SMOTE Confusion Matrix", use_container_width=True)
    else:
        st.info("Dataset not found in `data/`. Run `generate_dataset.py` to view live EDA.")

# ----------------------------------------
# TAB 3: MODEL INTELLIGENCE
# ----------------------------------------
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
    In our cross-validation trials across 5 distinct algorithms (including Decision Trees and Ensembles like Random Forest), Logistic Regression—when paired with SMOTE—yielded the **highest Macro Recall (73.14%)** and **Fatal Recall (78.80%)**. 
    
    *Translation for stakeholders: It minimizes False Negatives. It is the least likely to look at a highly dangerous intersection and accidentally declare it "Safe".*
    """)
