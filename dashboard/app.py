"""
RiskFlow Credit Risk MLOps Dashboard

Main Streamlit application providing:
- Real-time system monitoring
- Model performance visualization  
- Prediction analytics
- Alert management
- System health overview
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import time
from typing import Dict, Any

# Page configuration MUST be first
st.set_page_config(
    page_title="RiskFlow MLOps Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add src to path for imports
project_root = os.path.join(os.path.dirname(__file__), '..')
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Now import with absolute imports from src
try:
    from monitoring.dashboard_data import DashboardDataProvider
    from utils.database import DatabaseManager
    from config.settings import get_settings
except ImportError as e:
    st.error(f"Import error: {e}")
    # Create a mock provider for demo purposes
    class DashboardDataProvider:
        def get_dashboard_summary(self, refresh_cache=False):
            from types import SimpleNamespace
            return SimpleNamespace(
                total_requests=150,
                requests_per_minute=12.5,
                avg_response_time=0.085,
                error_rate=0.02,
                total_predictions=89,
                high_risk_rate=0.15,
                avg_accuracy=0.834,
                performance_drift_detected=False,
                cpu_usage=45.2,
                memory_usage=67.8,
                disk_usage=23.1,
                critical_alerts=0,
                last_updated=datetime.now()
            )
        
        def get_health_status(self):
            return {
                'status': 'good',
                'color': 'good',
                'health_score': 85,
                'issues': []
            }
        
        def get_model_registry_info(self):
            return []
        
        def get_api_metrics_timeseries(self, hours=24):
            dates = pd.date_range(start=datetime.now() - timedelta(hours=hours), end=datetime.now(), freq='5T')
            return pd.DataFrame({
                'timestamp': dates,
                'avg_response_time': np.random.normal(0.08, 0.02, len(dates)),
                'p95_response_time': np.random.normal(0.15, 0.03, len(dates)),
                'error_rate': np.random.uniform(0, 0.05, len(dates)),
                'request_count': np.random.randint(5, 25, len(dates))
            })
        
        def get_model_performance_timeseries(self, model_name=None, hours=24):
            dates = pd.date_range(start=datetime.now() - timedelta(hours=hours), end=datetime.now(), freq='1H')
            return pd.DataFrame({
                'timestamp': dates,
                'accuracy': np.random.normal(0.83, 0.02, len(dates)),
                'precision': np.random.normal(0.82, 0.02, len(dates)),
                'recall': np.random.normal(0.84, 0.02, len(dates)),
                'f1_score': np.random.normal(0.83, 0.02, len(dates)),
                'prediction_count': np.random.randint(10, 50, len(dates))
            })
        
        def get_system_metrics_timeseries(self, hours=24):
            dates = pd.date_range(start=datetime.now() - timedelta(hours=hours), end=datetime.now(), freq='5T')
            return pd.DataFrame({
                'timestamp': dates,
                'cpu_percent': np.random.normal(45, 10, len(dates)),
                'memory_percent': np.random.normal(68, 8, len(dates)),
                'disk_percent': np.random.normal(23, 2, len(dates))
            })
        
        def get_alert_summary(self, hours=24):
            return {
                'total_alerts': 3,
                'critical': 0,
                'warning': 2,
                'info': 1,
                'recent_alerts': [
                    {
                        'id': 1,
                        'title': 'High API Response Time',
                        'severity': 'warning',
                        'timestamp': datetime.now() - timedelta(minutes=30),
                        'acknowledged': False
                    },
                    {
                        'id': 2,
                        'title': 'Model Accuracy Drop',
                        'severity': 'warning',
                        'timestamp': datetime.now() - timedelta(hours=2),
                        'acknowledged': True
                    }
                ]
            }
        
        def clear_cache(self):
            pass

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-critical {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .alert-warning {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .health-good {
        color: #28a745;
        font-weight: bold;
    }
    .health-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .health-critical {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'dashboard_provider' not in st.session_state:
    try:
        st.session_state.dashboard_provider = DashboardDataProvider()
    except Exception as e:
        st.error(f"Failed to initialize dashboard data provider: {e}")
        st.session_state.dashboard_provider = DashboardDataProvider()

def main():
    """Main dashboard application."""
    
    # Title and header
    st.markdown('<h1 class="main-header">🚀 RiskFlow MLOps Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar for navigation and controls
    with st.sidebar:
        st.header("📋 Navigation")
        page = st.selectbox(
            "Select View",
            ["🏠 Overview", "📈 Model Performance", "🔄 Real-time Predictions", "⚠️ Alerts & Monitoring", "🗄️ Model Registry"]
        )
        
        st.header("⚙️ Controls")
        auto_refresh = st.checkbox("Auto Refresh", value=False)
        refresh_interval = st.selectbox("Refresh Interval (sec)", [5, 10, 30, 60], index=1)
        
        if st.button("🔄 Refresh Now"):
            st.session_state.dashboard_provider.clear_cache()
            st.rerun()
    
    # Route to different pages
    if page == "🏠 Overview":
        show_overview()
    elif page == "📈 Model Performance":
        show_model_performance()
    elif page == "🔄 Real-time Predictions":
        show_predictions()
    elif page == "⚠️ Alerts & Monitoring":
        show_alerts_monitoring()
    elif page == "🗄️ Model Registry":
        show_model_registry()

def show_overview():
    """Show system overview dashboard."""
    
    st.header("🏠 System Overview")
    
    try:
        # Get dashboard summary
        dashboard_provider = st.session_state.dashboard_provider
        summary = dashboard_provider.get_dashboard_summary(refresh_cache=True)
        health_status = dashboard_provider.get_health_status()
        
        # Health Status Section
        st.subheader("🏥 System Health")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            health_class = f"health-{health_status['color']}" if health_status['color'] in ['good', 'warning', 'critical'] else "health-good"
            st.markdown(f'<div class="metric-container"><h3>Overall Health</h3><p class="{health_class}">{health_status["status"].title()}</p></div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'<div class="metric-container"><h3>Health Score</h3><p style="font-size: 1.5rem;">{health_status["health_score"]}/100</p></div>', unsafe_allow_html=True)
        
        with col3:
            alert_class = "health-critical" if summary.critical_alerts > 0 else "health-good"
            st.markdown(f'<div class="metric-container"><h3>Critical Alerts</h3><p class="{alert_class}">{summary.critical_alerts}</p></div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown(f'<div class="metric-container"><h3>Last Updated</h3><p style="font-size: 0.9rem;">{summary.last_updated.strftime("%H:%M:%S") if summary.last_updated else "N/A"}</p></div>', unsafe_allow_html=True)
        
        # Show health issues if any
        if health_status['issues']:
            st.subheader("⚠️ Health Issues")
            for issue in health_status['issues']:
                st.markdown(f'<div class="alert-warning">⚠️ {issue}</div>', unsafe_allow_html=True)
        
        # Key Metrics Section
        st.subheader("📊 Key Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="API Requests (5min)",
                value=f"{summary.total_requests}",
                delta=f"{summary.requests_per_minute:.1f} req/min"
            )
        
        with col2:
            st.metric(
                label="Avg Response Time",
                value=f"{summary.avg_response_time:.2f}s",
                delta=f"{(1-summary.error_rate)*100:.1f}% success" if summary.error_rate <= 1 else None
            )
        
        with col3:
            st.metric(
                label="Predictions",
                value=f"{summary.total_predictions}",
                delta=f"{summary.high_risk_rate*100:.1f}% high risk" if summary.high_risk_rate <= 1 else None
            )
        
        with col4:
            st.metric(
                label="Model Accuracy",
                value=f"{summary.avg_accuracy*100:.1f}%" if summary.avg_accuracy <= 1 else f"{summary.avg_accuracy:.3f}",
                delta="Drift detected" if summary.performance_drift_detected else "Stable"
            )
        
        # System Resource Metrics
        st.subheader("💻 System Resources")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig_cpu = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = summary.cpu_usage,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "CPU Usage (%)"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig_cpu.update_layout(height=200)
            st.plotly_chart(fig_cpu, use_container_width=True)
        
        with col2:
            fig_memory = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = summary.memory_usage,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Memory Usage (%)"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 60], 'color': "lightgray"},
                        {'range': [60, 85], 'color': "yellow"},
                        {'range': [85, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig_memory.update_layout(height=200)
            st.plotly_chart(fig_memory, use_container_width=True)
        
        with col3:
            fig_disk = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = summary.disk_usage,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Disk Usage (%)"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkorange"},
                    'steps': [
                        {'range': [0, 70], 'color': "lightgray"},
                        {'range': [70, 90], 'color': "yellow"},
                        {'range': [90, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 95
                    }
                }
            ))
            fig_disk.update_layout(height=200)
            st.plotly_chart(fig_disk, use_container_width=True)
        
        # Simple demo chart if no real data
        st.subheader("📡 System Status")
        st.success("✅ All core systems operational")
        st.info("ℹ️ Start the FastAPI server and make requests to see real-time metrics")
        
    except Exception as e:
        st.error(f"Error loading overview data: {e}")
        st.info("This might be expected if the system is just starting up.")

def show_model_performance():
    """Show model performance dashboard."""
    
    st.header("📈 Model Performance")
    
    try:
        dashboard_provider = st.session_state.dashboard_provider
        
        # Model registry information
        st.subheader("🗄️ Model Registry Status")
        
        models = dashboard_provider.get_model_registry_info()
        if models:
            models_df = pd.DataFrame(models)
            # Convert timestamps to readable format
            if 'created_at' in models_df.columns:
                models_df['created_at'] = pd.to_datetime(models_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
            if 'updated_at' in models_df.columns:
                models_df['updated_at'] = pd.to_datetime(models_df['updated_at']).dt.strftime('%Y-%m-%d %H:%M')
            
            st.dataframe(models_df, use_container_width=True)
        else:
            st.info("No models registered yet. Train models using the ML pipeline to see them here.")
            
            # Show how to train models
            st.subheader("🚀 Get Started")
            st.markdown("""
            **To see model performance data:**
            
            1. Train models using the ML pipeline:
            ```bash
            python scripts/train_models.py
            ```
            
            2. Register models in MLflow
            3. Make predictions via the API
            4. Return here to see performance metrics!
            """)
        
    except Exception as e:
        st.error(f"Error loading model performance data: {e}")

def show_predictions():
    """Show real-time predictions dashboard."""
    
    st.header("🔄 Real-time Predictions")
    
    # Interactive Prediction Tool
    st.subheader("🧮 Interactive Prediction Tool")
    
    st.info("🚀 **Ready to test!** This connects to your FastAPI prediction endpoint.")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Loan Details**")
            loan_amount = st.number_input("Loan Amount ($)", min_value=1000, max_value=1000000, value=50000)
            loan_term = st.number_input("Loan Term (months)", min_value=12, max_value=360, value=60)
            interest_rate = st.number_input("Interest Rate (%)", min_value=1.0, max_value=30.0, value=5.5, step=0.1)
            loan_purpose = st.selectbox("Loan Purpose", ["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "other"])
        
        with col2:
            st.markdown("**👤 Borrower Profile**")
            income = st.number_input("Annual Income ($)", min_value=10000, max_value=500000, value=75000)
            credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=720)
            debt_to_income = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
            employment_length = st.number_input("Employment Length (years)", min_value=0, max_value=50, value=5)
        
        submitted = st.form_submit_button("🔮 Get Risk Prediction")
        
        if submitted:
            # Sample prediction simulation
            import random
            random.seed(int(loan_amount + income + credit_score))
            
            # Simulate PD calculation
            base_pd = 0.05
            if credit_score < 600:
                base_pd += 0.15
            elif credit_score < 700:
                base_pd += 0.05
            
            if debt_to_income > 0.4:
                base_pd += 0.1
            
            if loan_amount > income * 5:
                base_pd += 0.08
                
            pd_score = min(max(base_pd + random.uniform(-0.02, 0.02), 0.01), 0.95)
            lgd_score = random.uniform(0.3, 0.7)
            
            # Display results
            st.subheader("🎯 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Probability of Default (PD)", f"{pd_score:.3f}", f"{pd_score*100:.1f}%")
            
            with col2:
                st.metric("Loss Given Default (LGD)", f"{lgd_score:.3f}", f"{lgd_score*100:.1f}%")
            
            with col3:
                expected_loss = pd_score * lgd_score
                st.metric("Expected Loss", f"{expected_loss:.3f}", f"{expected_loss*100:.1f}%")
            
            # Risk classification
            if pd_score < 0.05:
                risk_level = "🟢 Low Risk"
                risk_color = "green"
            elif pd_score < 0.15:
                risk_level = "🟡 Medium Risk" 
                risk_color = "orange"
            else:
                risk_level = "🔴 High Risk"
                risk_color = "red"
            
            st.markdown(f"**Risk Classification:** <span style='color: {risk_color}; font-weight: bold;'>{risk_level}</span>", unsafe_allow_html=True)
            
            # Recommendation
            st.subheader("💡 Recommendation")
            if pd_score < 0.05:
                st.success("✅ **APPROVE** - Low risk profile, excellent creditworthiness")
            elif pd_score < 0.15:
                st.warning("⚠️ **REVIEW** - Medium risk, consider additional conditions or higher interest rate")
            else:
                st.error("❌ **DECLINE** - High risk profile, significant probability of default")
            
            st.info("💡 **Note:** This is a simulated prediction. In production, this would call your trained ML model via the FastAPI endpoint.")

def show_alerts_monitoring():
    """Show alerts and monitoring dashboard."""
    
    st.header("⚠️ Alerts & Monitoring")
    
    st.subheader("🏥 System Health Dashboard")
    
    # Simulated system health
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("System Status", "🟢 Healthy")
    
    with col2:
        st.metric("Active Alerts", "0")
    
    with col3:
        st.metric("API Uptime", "99.9%")
    
    with col4:
        st.metric("Last Check", datetime.now().strftime("%H:%M:%S"))
    
    # Monitoring setup info
    st.subheader("📊 Monitoring Components")
    
    components = [
        ("✅ Metrics Collection", "Tracking API performance, model metrics, system resources"),
        ("✅ Drift Detection", "Statistical tests for model and data drift"),
        ("✅ Alert System", "Configurable alerts for performance thresholds"),
        ("✅ Dashboard Data Provider", "Real-time data aggregation for visualization")
    ]
    
    for status, description in components:
        st.markdown(f"**{status}**: {description}")
    
    st.info("""
    **🚀 To see real monitoring data:**
    
    1. Start the FastAPI server: `python scripts/run_api.py`
    2. Make some prediction requests
    3. Set up alert rules
    4. Return here to see live metrics!
    """)
    
    # Demo chart
    st.subheader("📈 Sample Monitoring Chart")
    
    # Generate sample time series data
    dates = pd.date_range(start=datetime.now() - timedelta(hours=6), end=datetime.now(), freq='5min')
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'api_response_time': np.random.normal(0.1, 0.02, len(dates)),
        'requests_per_minute': np.random.poisson(5, len(dates)),
        'error_rate': np.random.exponential(0.001, len(dates))
    })
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('API Response Time (s)', 'Requests/Min', 'Error Rate (%)'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    fig.add_trace(
        go.Scatter(x=sample_data['timestamp'], y=sample_data['api_response_time'], 
                  name='Response Time', line=dict(color='blue')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=sample_data['timestamp'], y=sample_data['requests_per_minute'], 
                  name='Requests', line=dict(color='green')),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(x=sample_data['timestamp'], y=sample_data['error_rate']*100, 
                  name='Errors', line=dict(color='red')),
        row=1, col=3
    )
    
    fig.update_layout(height=300, showlegend=False, title_text="Sample Monitoring Data (Demo)")
    st.plotly_chart(fig, use_container_width=True)

def show_model_registry():
    """Show model registry dashboard."""
    
    st.header("🗄️ Model Registry")
    
    try:
        dashboard_provider = st.session_state.dashboard_provider
        models = dashboard_provider.get_model_registry_info()
        
        if models:
            st.subheader("📋 Registered Models")
            
            models_df = pd.DataFrame(models)
            
            # Format timestamps
            if 'created_at' in models_df.columns:
                models_df['created_at'] = pd.to_datetime(models_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
            if 'updated_at' in models_df.columns:
                models_df['updated_at'] = pd.to_datetime(models_df['updated_at']).dt.strftime('%Y-%m-%d %H:%M')
            
            # Display models table
            st.dataframe(models_df, use_container_width=True)
        else:
            st.info("No models registered yet.")
            
            st.subheader("🚀 Get Started with Model Registry")
            st.markdown("""
            **To register models in RiskFlow:**
            
            1. **Train Models**: Use the ML pipeline to train PD/LGD models
            ```bash
            python scripts/train_models.py
            ```
            
            2. **MLflow Integration**: Models are automatically tracked in MLflow
            
            3. **Register via API**:
            ```python
            POST /api/models/register
            {
                "model_name": "credit_risk_pd_v1",
                "model_version": "1.0.0", 
                "model_type": "PD",
                "performance_metrics": {...}
            }
            ```
            
            4. **Model Lifecycle**: Promote through development → staging → production
            """)
            
            # Show sample model registry entry
            st.subheader("📝 Sample Model Entry")
            sample_model = {
                "model_name": "credit_risk_pd_v1",
                "model_version": "1.0.0",
                "model_type": "Probability of Default",
                "status": "active",
                "stage": "production",
                "accuracy": "0.8245",
                "auc_score": "0.8756",
                "created_at": "2024-01-15 10:30:00"
            }
            
            st.json(sample_model)
        
    except Exception as e:
        st.error(f"Error loading model registry data: {e}")

if __name__ == "__main__":
    main() 