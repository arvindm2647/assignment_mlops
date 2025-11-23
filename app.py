import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
import json
import os
import sys

sys.path.append('src')

st.set_page_config(
    page_title="CPU Usage Prediction Dashboard",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css():
    try:
        if os.path.exists("styles.css"):
            with open("styles.css") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception as e:
        # CSS is optional, continue without it
        pass

load_css()

st.markdown('<p class="main-header">🖥️ CPU Usage Prediction Dashboard</p>', unsafe_allow_html=True)
st.markdown("---")

@st.cache_resource
def load_model():
    """Load the trained model, scaler, and metrics if they exist."""
    model_path = 'data/model.pkl'
    scaler_path = 'data/scaler.pkl'
    
    try:
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            metrics = {}
            if os.path.exists('metrics.json'):
                with open('metrics.json', 'r') as f:
                    metrics = json.load(f)
            return model, scaler, metrics
    except Exception as e:
        # Silently fail - model just doesn't exist yet
        return None, None, None
    
    return None, None, None

def train_model():
    """Train the model if it doesn't exist."""
    model_path = 'data/model.pkl'
    scaler_path = 'data/scaler.pkl'
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    if not os.path.exists('cpu_usage.csv'):
        st.error("❌ Error: cpu_usage.csv file not found! Please ensure the data file is in the repository.")
        return False
    
    try:
        from preprocess import preprocess
        from train import train
        from evaluate import evaluate
        
        with st.spinner("Step 1/3: Preprocessing data..."):
            preprocess()
        
        with st.spinner("Step 2/3: Training model..."):
            train()
        
        with st.spinner("Step 3/3: Evaluating model..."):
            evaluate()
        
        st.success("✅ Model trained successfully!")
        load_model.clear()  # Clear cache to reload model
        return True
    except Exception as e:
        st.error(f"❌ Training failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        return False

# Load model (cached)
model, scaler, metrics = load_model()

# If model doesn't exist, show training option
if model is None or scaler is None:
    st.warning("⚠️ Model not found. Click the button below to train the model.")
    if st.button("🚀 Train Model", type="primary"):
        if train_model():
            st.rerun()

st.sidebar.title("📋 Navigation")
page = st.sidebar.radio("Select Page", ["🏠 Home", "🎯 Prediction", "📊 Model Performance", "📈 Data Analysis"])

if page == "🏠 Home":
    st.header("📊 Project Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h3>🎯 Project Details</h3>
        <ul>
            <li><b>Assignment:</b> ML Deployment</li>
            <li><b>Model:</b> Random Forest</li>
            <li><b>Task:</b> CPU Usage Prediction</li>
            <li><b>Status:</b> ✅ Deployed</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h3>📈 Model Info</h3>
        <ul>
            <li><b>Type:</b> Regression</li>
            <li><b>Algorithm:</b> Random Forest</li>
            <li><b>Features:</b> 5 + Encoded</li>
            <li><b>Framework:</b> Scikit-learn</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h3>🚀 Features</h3>
        <ul>
            <li>Real-time Predictions</li>
            <li>Performance Metrics</li>
            <li>Interactive Visualizations</li>
            <li>Model Evaluation</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if metrics and len(metrics) > 0:
        st.subheader("📊 Model Performance Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Test Accuracy (R²)", 
                value=f"{metrics['test_r2']:.4f}",
                delta=f"{metrics['test_accuracy_percent']:.2f}%"
            )
        
        with col2:
            st.metric(
                label="Test RMSE", 
                value=f"{metrics['test_rmse']:.4f}"
            )
        
        with col3:
            st.metric(
                label="Test MAE", 
                value=f"{metrics['test_mae']:.4f}"
            )
        
        with col4:
            st.metric(
                label="Train Accuracy", 
                value=f"{metrics['train_r2']:.4f}"
            )
        
        st.markdown("---")
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Training',
            x=['R² Score', 'RMSE', 'MAE'],
            y=[metrics['train_r2'], metrics['train_rmse'], metrics['train_mae']],
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            name='Testing',
            x=['R² Score', 'RMSE', 'MAE'],
            y=[metrics['test_r2'], metrics['test_rmse'], metrics['test_mae']],
            marker_color='darkblue'
        ))
        
        fig.update_layout(
            title="Training vs Testing Metrics Comparison",
            xaxis_title="Metrics",
            yaxis_title="Value",
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, width='stretch')

elif page == "🎯 Prediction":
    st.header("🎯 Make Real-time Predictions")
    
    st.markdown("Enter the input features below to predict CPU usage.")
    
    if model and scaler:
        with st.form("prediction_form"):
            st.subheader("📝 Input Features")
            
            col1, col2 = st.columns(2)
            
            with col1:
                cpu_request = st.number_input(
                    "CPU Request", 
                    min_value=0.0, 
                    max_value=10.0,
                    value=0.5,
                    step=0.1
                )
                
                mem_request = st.number_input(
                    "Memory Request (GB)", 
                    min_value=0.0, 
                    max_value=100.0,
                    value=2.0,
                    step=0.5
                )
                
                cpu_limit = st.number_input(
                    "CPU Limit", 
                    min_value=0.0, 
                    max_value=10.0,
                    value=1.0,
                    step=0.1
                )
            
            with col2:
                mem_limit = st.number_input(
                    "Memory Limit (GB)", 
                    min_value=0.0, 
                    max_value=100.0,
                    value=4.0,
                    step=0.5
                )
                
                runtime_minutes = st.number_input(
                    "Runtime (minutes)", 
                    min_value=0.0, 
                    max_value=10000.0,
                    value=60.0,
                    step=10.0
                )
                
                controller_kind = st.selectbox(
                    "Controller Kind",
                    options=["DaemonSet", "Job", "ReplicaSet", "ReplicationController", "StatefulSet"]
                )
            
            submitted = st.form_submit_button("🔮 Predict CPU Usage")
            
            if submitted:
                try:
                    input_data = pd.DataFrame({
                        'cpu_request': [cpu_request],
                        'mem_request': [mem_request],
                        'cpu_limit': [cpu_limit],
                        'mem_limit': [mem_limit],
                        'runtime_minutes': [runtime_minutes]
                    })
                    
                    controller_options = ["DaemonSet", "Job", "ReplicaSet", "ReplicationController", "StatefulSet"]
                    for option in controller_options:
                        input_data[f'controller_kind_{option}'] = 1 if controller_kind == option else 0
                    
                    numeric_features = ['cpu_request', 'mem_request', 'cpu_limit', 'mem_limit', 'runtime_minutes']
                    input_data[numeric_features] = scaler.transform(input_data[numeric_features])
                    
                    prediction = model.predict(input_data)[0]
                    
                    st.success("✅ Prediction Complete!")
                    
                    st.markdown("---")
                    
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col2:
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h2>Predicted CPU Usage</h2>
                            <h1>{prediction:.4f}</h1>
                            <p>CPU Cores</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prediction,
                        title={'text': "CPU Usage Prediction"},
                        gauge={
                            'axis': {'range': [0, max(2, prediction * 1.5)]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, prediction * 0.5], 'color': "lightgreen"},
                                {'range': [prediction * 0.5, prediction], 'color': "lightyellow"},
                                {'range': [prediction, prediction * 1.5], 'color': "lightcoral"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': prediction
                            }
                        }
                    ))
                    
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, width='stretch')
                    
                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")
    else:
        st.error("❌ Model or scaler not loaded properly!")

elif page == "📊 Model Performance":
    st.header("📊 Detailed Model Performance")
    
    if metrics and len(metrics) > 0:
        st.subheader("📋 Performance Metrics Table")
        
        metrics_df = pd.DataFrame({
            'Metric': ['MSE', 'RMSE', 'MAE', 'R² Score', 'Accuracy %'],
            'Training': [
                f"{metrics['train_mse']:.6f}",
                f"{metrics['train_rmse']:.6f}",
                f"{metrics['train_mae']:.6f}",
                f"{metrics['train_r2']:.6f}",
                f"{metrics['train_accuracy_percent']:.2f}%"
            ],
            'Testing': [
                f"{metrics['test_mse']:.6f}",
                f"{metrics['test_rmse']:.6f}",
                f"{metrics['test_mae']:.6f}",
                f"{metrics['test_r2']:.6f}",
                f"{metrics['test_accuracy_percent']:.2f}%"
            ]
        })
        
        st.dataframe(metrics_df, width='stretch', hide_index=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Training Performance")
            fig1 = go.Figure()
            fig1.add_trace(go.Bar(
                x=['RMSE', 'MAE', 'R²'],
                y=[metrics['train_rmse'], metrics['train_mae'], metrics['train_r2']],
                marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1']
            ))
            fig1.update_layout(title="Training Metrics", height=400)
            st.plotly_chart(fig1, width='stretch')
        
        with col2:
            st.subheader("🎯 Testing Performance")
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=['RMSE', 'MAE', 'R²'],
                y=[metrics['test_rmse'], metrics['test_mae'], metrics['test_r2']],
                marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1']
            ))
            fig2.update_layout(title="Testing Metrics", height=400)
            st.plotly_chart(fig2, width='stretch')
        
        st.markdown("---")
        
        st.subheader("📈 R² Score Comparison")
        fig3 = go.Figure()
        fig3.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=metrics['test_r2'],
            title={'text': "Test R² Score"},
            delta={'reference': metrics['train_r2']},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.5], 'color': "lightgray"},
                    {'range': [0.5, 0.8], 'color': "lightyellow"},
                    {'range': [0.8, 1], 'color': "lightgreen"}
                ]
            }
        ))
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, width='stretch')

elif page == "📈 Data Analysis":
    st.header("📈 Data Analysis & Insights")
    
    try:
        df = pd.read_csv('data/processed_data.csv')
        
        st.subheader("📊 Dataset Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            st.metric("Features", len(df.columns) - 1)
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        st.markdown("---")
        
        st.subheader("🔍 Data Preview")
        st.dataframe(df.head(10), width='stretch')
        
        st.markdown("---")
        
        st.subheader("📊 Statistical Summary")
        st.dataframe(df.describe(), width='stretch')
        
        st.markdown("---")
        
        st.subheader("📉 CPU Usage Distribution")
        fig = px.histogram(df, x='cpu_usage', nbins=50, title="Distribution of CPU Usage")
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')
        
    except Exception as e:
        st.error(f"Error loading data: {e}")

st.markdown("---")
st.markdown("""
<div class="footer">
    <p>🖥️ CPU Usage Prediction Dashboard | Built with Streamlit & Scikit-learn</p>
    <p>Assignment 2: ML Model Deployment</p>
</div>
""", unsafe_allow_html=True)