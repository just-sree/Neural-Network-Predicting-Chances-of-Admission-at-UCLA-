import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import os

# Page config
st.set_page_config(
    page_title="🎓 UCLA Admission Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        height: 3em;
        margin-top: 1em;
    }
    .stat-box {
        padding: 1em;
        border-radius: 5px;
        border: 2px solid #f0f2f6;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def create_gauge_chart(value, title, max_val):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, max_val]},
            'bar': {'color': "#2E86C1"},
            'steps': [
                {'range': [0, max_val/3], 'color': "#FF9999"},
                {'range': [max_val/3, 2*max_val/3], 'color': "#FFE699"},
                {'range': [2*max_val/3, max_val], 'color': "#99FF99"}
            ]
        }
    ))
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=40, b=10))
    return fig

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    show_feature_importance = st.checkbox("Show Feature Importance", True)
    show_probability = st.checkbox("Show Probability Score", True)
    show_metrics = st.checkbox("Show Score Metrics", True)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.title("🎓 UCLA Admission Predictor")
    st.markdown("### Enter your academic profile:")

    # Load model
    model_path = 'models/admission_nn_model.pkl'
    if not os.path.exists(model_path):
        st.error("❌ Model not found. Please train the model using `train.py`.")
        st.stop()

    model = joblib.load(model_path)

    # Input form with columns
    col1_form, col2_form = st.columns(2)

    with col1_form:
        gre = st.slider("GRE Score", 260, 340, 310, help="Graduate Record Examination score")
        toefl = st.slider("TOEFL Score", 0, 120, 100, help="Test of English as a Foreign Language score")
        rating = st.selectbox("University Rating", [1, 2, 3, 4, 5], index=2, 
                            help="Previous university's rating (1 lowest, 5 highest)")

    with col2_form:
        sop = st.slider("SOP Strength", 1.0, 5.0, 3.0, step=0.5, 
                       help="Statement of Purpose strength")
        lor = st.slider("LOR Strength", 1.0, 5.0, 3.0, step=0.5, 
                       help="Letter of Recommendation strength")
        cgpa = st.slider("CGPA", 6.0, 10.0, 8.5, step=0.1, 
                        help="Cumulative Grade Point Average")

    research = st.radio("Research Experience", ["Yes", "No"], horizontal=True) == "Yes"

    # Prepare input
    input_dict = {
        "GRE_Score": gre,
        "TOEFL_Score": toefl,
        "SOP": sop,
        "LOR": lor,
        "CGPA": cgpa,
        "University_Rating_2": 1 if rating == 2 else 0,
        "University_Rating_3": 1 if rating == 3 else 0,
        "University_Rating_4": 1 if rating == 4 else 0,
        "University_Rating_5": 1 if rating == 5 else 0,
        "Research_1": 1 if research else 0
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

with col2:
    st.markdown("### 📊 Score Overview")
    
    # Create gauge charts for main scores
    st.plotly_chart(create_gauge_chart(gre, "GRE Score", 340))
    st.plotly_chart(create_gauge_chart(toefl, "TOEFL Score", 120))
    st.plotly_chart(create_gauge_chart(cgpa * 10, "CGPA %", 100))

if st.button("🔮 Predict Admission Chances"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    
    # Result section
    st.markdown("---")
    result_col1, result_col2 = st.columns([2, 1])
    
    with result_col1:
        st.markdown("### 📊 Prediction Results")
        
        # Probability gauge
        if show_probability:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob * 100,
                title={'text': "Admission Probability"},
                domain={'x': [0, 1], 'y': [0, 1]},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#2E86C1"},
                    'steps': [
                        {'range': [0, 30], 'color': "#FF9999"},
                        {'range': [30, 70], 'color': "#FFE699"},
                        {'range': [70, 100], 'color': "#99FF99"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        # Decision
        if prob > 0.5:
            st.success("🎉 High Chance of Admission!")
        else:
            st.error("❌ Low Chance of Admission")

    with result_col2:
        if show_metrics:
            st.markdown("### 📈 Key Metrics")
            metrics = {
                "GRE": f"{gre}/340",
                "TOEFL": f"{toefl}/120",
                "CGPA": f"{cgpa}/10",
                "Research": "Yes" if research else "No"
            }
            
            for metric, value in metrics.items():
                st.markdown(f"""
                <div class="stat-box">
                    <h3>{metric}</h3>
                    <h2>{value}</h2>
                </div>
                """, unsafe_allow_html=True)

    # Feature importance visualization
    if show_feature_importance:
        st.markdown("### 🎯 Feature Contribution")
        feature_importance = pd.DataFrame({
            'Feature': input_df.columns,
            'Value': input_df.iloc[0].values
        })
        fig = px.bar(feature_importance, x='Value', y='Feature', 
                    orientation='h', title='Your Profile Breakdown')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)