"""
Reusable Chart Components for RiskFlow Dashboard

Standard chart configurations and styling for consistent visualization.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

# Standard RiskFlow color palette
RISKFLOW_COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e', 
    'success': '#2ca02c',
    'warning': '#d62728',
    'info': '#9467bd',
    'light': '#8c564b',
    'dark': '#e377c2'
}

def create_gauge_chart(
    value: float,
    title: str,
    max_value: float = 100,
    thresholds: Dict[str, float] = None,
    colors: Dict[str, str] = None
) -> go.Figure:
    """
    Create a standardized gauge chart.
    
    Args:
        value: Current value to display
        title: Chart title
        max_value: Maximum value for the gauge
        thresholds: Dict with 'warning' and 'critical' thresholds
        colors: Dict with custom colors
    
    Returns:
        Plotly Figure object
    """
    if thresholds is None:
        thresholds = {'warning': max_value * 0.7, 'critical': max_value * 0.9}
    
    if colors is None:
        colors = {
            'bar': RISKFLOW_COLORS['primary'],
            'good': 'lightgray',
            'warning': '#ffc107',
            'critical': '#dc3545'
        }
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [None, max_value]},
            'bar': {'color': colors['bar']},
            'steps': [
                {'range': [0, thresholds['warning']], 'color': colors['good']},
                {'range': [thresholds['warning'], thresholds['critical']], 'color': colors['warning']},
                {'range': [thresholds['critical'], max_value], 'color': colors['critical']}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': thresholds['critical']
            }
        }
    ))
    
    fig.update_layout(height=200, font={'size': 12})
    return fig

def create_time_series_chart(
    data: pd.DataFrame,
    x_col: str,
    y_cols: List[str],
    title: str,
    labels: Dict[str, str] = None,
    colors: List[str] = None
) -> go.Figure:
    """
    Create a standardized time series chart.
    
    Args:
        data: DataFrame with time series data
        x_col: Column name for x-axis (time)
        y_cols: List of column names for y-axis
        title: Chart title
        labels: Dict for custom axis labels
        colors: List of custom colors for each series
    
    Returns:
        Plotly Figure object
    """
    if colors is None:
        colors = [RISKFLOW_COLORS['primary'], RISKFLOW_COLORS['secondary'], 
                 RISKFLOW_COLORS['success'], RISKFLOW_COLORS['warning']]
    
    fig = go.Figure()
    
    for i, col in enumerate(y_cols):
        if col in data.columns:
            fig.add_trace(go.Scatter(
                x=data[x_col],
                y=data[col],
                mode='lines+markers',
                name=col.replace('_', ' ').title(),
                line={'color': colors[i % len(colors)]}
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title=labels.get('x', x_col) if labels else x_col,
        yaxis_title=labels.get('y', 'Value') if labels else 'Value',
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_distribution_chart(
    data: List[float],
    title: str,
    bins: int = 20,
    color: str = None
) -> go.Figure:
    """
    Create a histogram/distribution chart.
    
    Args:
        data: List of values to plot
        title: Chart title
        bins: Number of bins for histogram
        color: Custom color for bars
    
    Returns:
        Plotly Figure object
    """
    if color is None:
        color = RISKFLOW_COLORS['primary']
    
    fig = go.Figure(data=[go.Histogram(
        x=data,
        nbinsx=bins,
        marker_color=color,
        opacity=0.7
    )])
    
    fig.update_layout(
        title=title,
        xaxis_title="Value",
        yaxis_title="Frequency",
        height=400
    )
    
    return fig

def create_metrics_dashboard(
    metrics_data: Dict[str, Any],
    layout: str = "2x2"
) -> go.Figure:
    """
    Create a multi-panel metrics dashboard.
    
    Args:
        metrics_data: Dict with metric names as keys and data as values
        layout: Layout pattern (e.g., "2x2", "1x3", "2x3")
    
    Returns:
        Plotly Figure object with subplots
    """
    # Parse layout
    rows, cols = map(int, layout.split('x'))
    
    # Create subplot titles
    subplot_titles = list(metrics_data.keys())
    
    fig = make_subplots(
        rows=rows, 
        cols=cols,
        subplot_titles=subplot_titles,
        specs=[[{"secondary_y": False} for _ in range(cols)] for _ in range(rows)]
    )
    
    # Add traces
    for i, (metric_name, metric_data) in enumerate(metrics_data.items()):
        row = (i // cols) + 1
        col = (i % cols) + 1
        
        if isinstance(metric_data, pd.DataFrame) and 'timestamp' in metric_data.columns:
            # Time series data
            y_col = [col for col in metric_data.columns if col != 'timestamp'][0]
            fig.add_trace(
                go.Scatter(
                    x=metric_data['timestamp'],
                    y=metric_data[y_col],
                    mode='lines',
                    name=metric_name,
                    line={'color': RISKFLOW_COLORS['primary']}
                ),
                row=row, col=col
            )
        elif isinstance(metric_data, list):
            # Simple line plot
            fig.add_trace(
                go.Scatter(
                    y=metric_data,
                    mode='lines+markers',
                    name=metric_name,
                    line={'color': RISKFLOW_COLORS['primary']}
                ),
                row=row, col=col
            )
    
    fig.update_layout(height=400, showlegend=False)
    return fig

def create_status_indicator(
    status: str,
    message: str = None,
    size: str = "medium"
) -> str:
    """
    Create a status indicator with emoji and color coding.
    
    Args:
        status: Status level ('good', 'warning', 'critical', 'info')
        message: Optional message to display
        size: Size of the indicator ('small', 'medium', 'large')
    
    Returns:
        HTML string for status indicator
    """
    status_config = {
        'good': {'emoji': '🟢', 'color': '#28a745', 'text': 'Good'},
        'warning': {'emoji': '🟡', 'color': '#ffc107', 'text': 'Warning'},
        'critical': {'emoji': '🔴', 'color': '#dc3545', 'text': 'Critical'},
        'info': {'emoji': '🔵', 'color': '#17a2b8', 'text': 'Info'}
    }
    
    size_config = {
        'small': '1rem',
        'medium': '1.2rem',
        'large': '1.5rem'
    }
    
    config = status_config.get(status, status_config['info'])
    font_size = size_config.get(size, size_config['medium'])
    
    display_text = message or config['text']
    
    return f"""
    <div style="display: flex; align-items: center; font-size: {font_size};">
        <span style="margin-right: 0.5rem;">{config['emoji']}</span>
        <span style="color: {config['color']}; font-weight: bold;">{display_text}</span>
    </div>
    """

def create_performance_heatmap(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    value_col: str,
    title: str
) -> go.Figure:
    """
    Create a performance heatmap for visualizing model metrics.
    
    Args:
        data: DataFrame with performance data
        x_col: Column for x-axis
        y_col: Column for y-axis  
        value_col: Column for heatmap values
        title: Chart title
    
    Returns:
        Plotly Figure object
    """
    # Pivot data for heatmap
    heatmap_data = data.pivot(index=y_col, columns=x_col, values=value_col)
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='RdYlBu_r',
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_col,
        height=400
    )
    
    return fig 