"""
Reusable Metrics Components for RiskFlow Dashboard

Standard metric displays and calculations for consistent presentation.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta

def display_key_metrics(
    metrics: Dict[str, Union[int, float, str]],
    columns: int = 4,
    title: str = None
) -> None:
    """
    Display key metrics in a grid layout.
    
    Args:
        metrics: Dictionary of metric names and values
        columns: Number of columns in the grid
        title: Optional section title
    """
    if title:
        st.subheader(title)
    
    # Create columns
    cols = st.columns(columns)
    
    for i, (metric_name, value) in enumerate(metrics.items()):
        col_idx = i % columns
        
        with cols[col_idx]:
            # Format the metric name
            display_name = metric_name.replace('_', ' ').title()
            
            # Format the value based on type
            if isinstance(value, float):
                if 0 < value < 1:
                    # Percentage or rate
                    display_value = f"{value:.2%}" if value <= 1 else f"{value:.4f}"
                else:
                    display_value = f"{value:.2f}"
            elif isinstance(value, int):
                display_value = f"{value:,}"
            else:
                display_value = str(value)
            
            st.metric(label=display_name, value=display_value)

def display_comparison_metrics(
    current_metrics: Dict[str, Union[int, float]],
    previous_metrics: Dict[str, Union[int, float]] = None,
    columns: int = 4,
    title: str = None
) -> None:
    """
    Display metrics with comparison to previous period.
    
    Args:
        current_metrics: Current period metrics
        previous_metrics: Previous period metrics for comparison
        columns: Number of columns in the grid
        title: Optional section title
    """
    if title:
        st.subheader(title)
    
    cols = st.columns(columns)
    
    for i, (metric_name, current_value) in enumerate(current_metrics.items()):
        col_idx = i % columns
        
        with cols[col_idx]:
            display_name = metric_name.replace('_', ' ').title()
            
            # Format current value
            if isinstance(current_value, float):
                if 0 < current_value < 1:
                    display_value = f"{current_value:.2%}" if current_value <= 1 else f"{current_value:.4f}"
                else:
                    display_value = f"{current_value:.2f}"
            else:
                display_value = f"{current_value:,}"
            
            # Calculate delta if previous value exists
            delta = None
            if previous_metrics and metric_name in previous_metrics:
                previous_value = previous_metrics[metric_name]
                if isinstance(current_value, (int, float)) and isinstance(previous_value, (int, float)):
                    if previous_value != 0:
                        delta_pct = ((current_value - previous_value) / previous_value) * 100
                        delta = f"{delta_pct:+.1f}%"
                    else:
                        delta = "New"
            
            st.metric(label=display_name, value=display_value, delta=delta)

def calculate_system_health_score(metrics: Dict[str, float]) -> Dict[str, Any]:
    """
    Calculate overall system health score.
    
    Args:
        metrics: Dictionary of system metrics
        
    Returns:
        Health score and status information
    """
    health_components = {
        'api_response_time': {'weight': 0.2, 'target': 0.1, 'critical': 1.0, 'lower_is_better': True},
        'error_rate': {'weight': 0.25, 'target': 0.01, 'critical': 0.05, 'lower_is_better': True},
        'cpu_usage': {'weight': 0.15, 'target': 50, 'critical': 90, 'lower_is_better': True},
        'memory_usage': {'weight': 0.15, 'target': 60, 'critical': 85, 'lower_is_better': True},
        'model_accuracy': {'weight': 0.25, 'target': 0.85, 'critical': 0.70, 'lower_is_better': False}
    }
    
    total_score = 0
    component_scores = {}
    issues = []
    
    for metric_name, config in health_components.items():
        if metric_name in metrics:
            value = metrics[metric_name]
            target = config['target']
            critical = config['critical']
            weight = config['weight']
            lower_is_better = config['lower_is_better']
            
            # Calculate component score (0-100)
            if lower_is_better:
                if value <= target:
                    score = 100
                elif value >= critical:
                    score = 0
                else:
                    score = 100 * (critical - value) / (critical - target)
                
                if value >= critical:
                    issues.append(f"{metric_name.replace('_', ' ').title()} is critical ({value:.2f})")
                elif value >= target:
                    issues.append(f"{metric_name.replace('_', ' ').title()} is above target ({value:.2f})")
            else:
                if value >= target:
                    score = 100
                elif value <= critical:
                    score = 0
                else:
                    score = 100 * (value - critical) / (target - critical)
                
                if value <= critical:
                    issues.append(f"{metric_name.replace('_', ' ').title()} is critical ({value:.2f})")
                elif value <= target:
                    issues.append(f"{metric_name.replace('_', ' ').title()} is below target ({value:.2f})")
            
            component_scores[metric_name] = score
            total_score += score * weight
    
    # Determine overall status
    if total_score >= 90:
        status = "excellent"
        color = "good"
    elif total_score >= 75:
        status = "good"
        color = "good"
    elif total_score >= 60:
        status = "warning"
        color = "warning"
    else:
        status = "critical"
        color = "critical"
    
    return {
        'health_score': int(total_score),
        'status': status,
        'color': color,
        'component_scores': component_scores,
        'issues': issues
    }

def display_health_indicators(
    health_data: Dict[str, Any],
    show_details: bool = True
) -> None:
    """
    Display system health indicators.
    
    Args:
        health_data: Health data from calculate_system_health_score
        show_details: Whether to show detailed component scores
    """
    # Main health score
    col1, col2, col3 = st.columns(3)
    
    with col1:
        score = health_data['health_score']
        color = '🟢' if score >= 75 else '🟡' if score >= 60 else '🔴'
        st.metric("System Health", f"{color} {score}/100")
    
    with col2:
        status = health_data['status'].title()
        st.metric("Status", status)
    
    with col3:
        issues_count = len(health_data.get('issues', []))
        st.metric("Active Issues", issues_count)
    
    # Show issues if any
    if health_data.get('issues'):
        st.subheader("⚠️ Health Issues")
        for issue in health_data['issues']:
            if 'critical' in issue.lower():
                st.error(f"🔴 {issue}")
            else:
                st.warning(f"🟡 {issue}")
    
    # Show component details
    if show_details and health_data.get('component_scores'):
        st.subheader("📊 Component Health")
        
        component_cols = st.columns(len(health_data['component_scores']))
        
        for i, (component, score) in enumerate(health_data['component_scores'].items()):
            with component_cols[i]:
                component_name = component.replace('_', ' ').title()
                color = '🟢' if score >= 75 else '🟡' if score >= 60 else '🔴'
                st.metric(component_name, f"{color} {int(score)}")

def format_performance_metrics(metrics: Dict[str, float]) -> Dict[str, str]:
    """
    Format performance metrics for display.
    
    Args:
        metrics: Raw performance metrics
        
    Returns:
        Formatted metrics dictionary
    """
    formatted = {}
    
    for key, value in metrics.items():
        if key in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_score']:
            # Performance scores (0-1)
            formatted[key] = f"{value:.4f} ({value*100:.1f}%)"
        elif key in ['response_time', 'prediction_time']:
            # Time metrics
            if value < 1:
                formatted[key] = f"{value*1000:.1f}ms"
            else:
                formatted[key] = f"{value:.2f}s"
        elif key in ['error_rate', 'drift_rate']:
            # Rates (0-1)
            formatted[key] = f"{value*100:.2f}%"
        elif key in ['cpu_usage', 'memory_usage', 'disk_usage']:
            # Usage percentages
            formatted[key] = f"{value:.1f}%"
        elif key in ['request_count', 'prediction_count']:
            # Counts
            formatted[key] = f"{int(value):,}"
        else:
            # Default formatting
            if isinstance(value, float):
                formatted[key] = f"{value:.4f}"
            else:
                formatted[key] = str(value)
    
    return formatted

def create_metric_card(
    title: str,
    value: Union[int, float, str],
    delta: Union[int, float, str] = None,
    delta_color: str = "normal",
    help_text: str = None
) -> None:
    """
    Create a styled metric card.
    
    Args:
        title: Metric title
        value: Metric value
        delta: Change from previous period
        delta_color: Color for delta ("normal", "inverse")
        help_text: Optional help text
    """
    # Custom CSS for metric card
    card_style = """
    <div style="
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    ">
        <h4 style="margin: 0; font-size: 1rem; color: #333;">{title}</h4>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.5rem; font-weight: bold; color: #1f77b4;">{value}</p>
        {delta_html}
        {help_html}
    </div>
    """
    
    # Format delta
    delta_html = ""
    if delta is not None:
        delta_str = str(delta)
        if delta_str.startswith(('+', '-')):
            color = "#28a745" if delta_str.startswith('+') else "#dc3545"
            if delta_color == "inverse":
                color = "#dc3545" if delta_str.startswith('+') else "#28a745"
            delta_html = f'<p style="margin: 0; font-size: 0.9rem; color: {color};">{delta_str}</p>'
    
    # Format help text
    help_html = ""
    if help_text:
        help_html = f'<p style="margin: 0.5rem 0 0 0; font-size: 0.8rem; color: #666; font-style: italic;">{help_text}</p>'
    
    # Render card
    st.markdown(
        card_style.format(
            title=title,
            value=value,
            delta_html=delta_html,
            help_html=help_html
        ),
        unsafe_allow_html=True
    )

def calculate_model_performance_summary(performance_data: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate summary statistics for model performance.
    
    Args:
        performance_data: List of performance records
        
    Returns:
        Summary statistics
    """
    if not performance_data:
        return {}
    
    df = pd.DataFrame(performance_data)
    
    summary = {}
    
    # Performance metrics
    for metric in ['accuracy', 'precision_score', 'recall_score', 'f1_score', 'auc_score']:
        if metric in df.columns:
            summary[f'avg_{metric}'] = df[metric].mean()
            summary[f'min_{metric}'] = df[metric].min()
            summary[f'max_{metric}'] = df[metric].max()
            summary[f'std_{metric}'] = df[metric].std()
    
    # Prediction statistics
    if 'prediction_count' in df.columns:
        summary['total_predictions'] = df['prediction_count'].sum()
        summary['avg_predictions_per_period'] = df['prediction_count'].mean()
    
    if 'avg_prediction_time' in df.columns:
        summary['avg_prediction_time'] = df['avg_prediction_time'].mean()
    
    # Risk distribution
    if 'avg_pd_score' in df.columns:
        summary['avg_pd_score'] = df['avg_pd_score'].mean()
        summary['high_risk_rate'] = (df['avg_pd_score'] > 0.15).mean()
    
    return summary 