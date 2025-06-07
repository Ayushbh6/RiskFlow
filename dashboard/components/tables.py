"""
Reusable Table Components for RiskFlow Dashboard

Standard table formatting and utilities for data display.
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Optional, Callable

def format_model_registry_table(models_df: pd.DataFrame) -> pd.DataFrame:
    """
    Format model registry data for display.
    
    Args:
        models_df: DataFrame with model registry data
        
    Returns:
        Formatted DataFrame
    """
    if models_df.empty:
        return models_df
    
    # Create a copy to avoid modifying original
    formatted_df = models_df.copy()
    
    # Format timestamps
    for col in ['created_at', 'updated_at', 'last_used']:
        if col in formatted_df.columns:
            formatted_df[col] = pd.to_datetime(formatted_df[col]).dt.strftime('%Y-%m-%d %H:%M')
    
    # Format performance metrics
    for col in ['accuracy', 'precision_score', 'recall_score', 'f1_score', 'auc_score']:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
    
    # Reorder columns for better display
    preferred_order = [
        'model_name', 'model_version', 'model_type', 'status', 'stage',
        'accuracy', 'auc_score', 'created_at', 'updated_at'
    ]
    
    # Only include columns that exist
    display_columns = [col for col in preferred_order if col in formatted_df.columns]
    remaining_columns = [col for col in formatted_df.columns if col not in display_columns]
    
    return formatted_df[display_columns + remaining_columns]

def format_prediction_log_table(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Format prediction log data for display.
    
    Args:
        predictions_df: DataFrame with prediction log data
        
    Returns:
        Formatted DataFrame
    """
    if predictions_df.empty:
        return predictions_df
    
    formatted_df = predictions_df.copy()
    
    # Format timestamp
    if 'timestamp' in formatted_df.columns:
        formatted_df['timestamp'] = pd.to_datetime(formatted_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Format prediction scores
    for col in ['pd_score', 'lgd_score', 'ead_score']:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
    
    # Format prediction time
    if 'prediction_time' in formatted_df.columns:
        formatted_df['prediction_time'] = formatted_df['prediction_time'].apply(
            lambda x: f"{x:.3f}s" if pd.notnull(x) else "N/A"
        )
    
    # Add risk level based on PD score
    if 'pd_score' in formatted_df.columns:
        formatted_df['risk_level'] = formatted_df['pd_score'].apply(classify_risk_level)
    
    return formatted_df

def format_alert_table(alerts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Format alerts data for display.
    
    Args:
        alerts_df: DataFrame with alerts data
        
    Returns:
        Formatted DataFrame with styling
    """
    if alerts_df.empty:
        return alerts_df
    
    formatted_df = alerts_df.copy()
    
    # Format timestamp
    if 'timestamp' in formatted_df.columns:
        formatted_df['timestamp'] = pd.to_datetime(formatted_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Add emoji indicators for severity
    if 'severity' in formatted_df.columns:
        severity_map = {
            'critical': '🔴 Critical',
            'high': '🟠 High',
            'medium': '🟡 Medium',
            'low': '🟢 Low',
            'info': '🔵 Info'
        }
        formatted_df['severity'] = formatted_df['severity'].map(severity_map)
    
    # Format resolved status
    if 'resolved' in formatted_df.columns:
        formatted_df['status'] = formatted_df['resolved'].apply(
            lambda x: '✅ Resolved' if x else '🔴 Active'
        )
        formatted_df = formatted_df.drop('resolved', axis=1)
    
    return formatted_df

def classify_risk_level(pd_score: float) -> str:
    """
    Classify risk level based on PD score.
    
    Args:
        pd_score: Probability of Default score
        
    Returns:
        Risk level string with emoji
    """
    if pd.isna(pd_score):
        return "❓ Unknown"
    
    if pd_score < 0.05:
        return "🟢 Low"
    elif pd_score < 0.15:
        return "🟡 Medium"
    elif pd_score < 0.30:
        return "🟠 High"
    else:
        return "🔴 Critical"

def create_sortable_table(
    data: pd.DataFrame,
    title: str = None,
    key: str = None,
    hide_index: bool = True,
    height: int = None
) -> None:
    """
    Create a sortable, filterable table with Streamlit.
    
    Args:
        data: DataFrame to display
        title: Optional table title
        key: Unique key for the table widget
        hide_index: Whether to hide the DataFrame index
        height: Optional fixed height for the table
    """
    if title:
        st.subheader(title)
    
    if data.empty:
        st.info("No data available")
        return
    
    # Add column filters if data is large
    if len(data) > 20:
        with st.expander("🔍 Filters", expanded=False):
            filter_columns = st.multiselect(
                "Filter by columns:",
                options=data.columns.tolist(),
                key=f"filter_{key}" if key else None
            )
            
            if filter_columns:
                # Create filters for selected columns
                filtered_data = data.copy()
                for col in filter_columns:
                    if data[col].dtype == 'object':
                        # String column - multiselect filter
                        unique_values = data[col].unique().tolist()
                        selected_values = st.multiselect(
                            f"Filter {col}:",
                            options=unique_values,
                            default=unique_values,
                            key=f"filter_{col}_{key}" if key else None
                        )
                        filtered_data = filtered_data[filtered_data[col].isin(selected_values)]
                    else:
                        # Numeric column - range filter
                        min_val, max_val = float(data[col].min()), float(data[col].max())
                        selected_range = st.slider(
                            f"Filter {col}:",
                            min_value=min_val,
                            max_value=max_val,
                            value=(min_val, max_val),
                            key=f"filter_{col}_{key}" if key else None
                        )
                        filtered_data = filtered_data[
                            (filtered_data[col] >= selected_range[0]) & 
                            (filtered_data[col] <= selected_range[1])
                        ]
                
                data = filtered_data
    
    # Display table
    st.dataframe(
        data,
        use_container_width=True,
        hide_index=hide_index,
        height=height,
        key=key
    )
    
    # Display summary statistics
    st.caption(f"Showing {len(data):,} records")

def create_metrics_table(
    metrics: Dict[str, Any],
    title: str = "Metrics Summary"
) -> None:
    """
    Create a formatted metrics table.
    
    Args:
        metrics: Dictionary of metric names and values
        title: Table title
    """
    st.subheader(title)
    
    # Convert metrics to DataFrame
    metrics_data = []
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
        else:
            formatted_value = str(value)
        
        metrics_data.append({
            'Metric': key.replace('_', ' ').title(),
            'Value': formatted_value
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Style the table
    st.dataframe(
        metrics_df,
        use_container_width=True,
        hide_index=True
    )

def create_comparison_table(
    data_dict: Dict[str, pd.DataFrame],
    title: str = "Comparison Table"
) -> None:
    """
    Create a side-by-side comparison table.
    
    Args:
        data_dict: Dictionary with labels as keys and DataFrames as values
        title: Table title
    """
    st.subheader(title)
    
    if not data_dict:
        st.info("No data for comparison")
        return
    
    # Create columns for each DataFrame
    cols = st.columns(len(data_dict))
    
    for i, (label, data) in enumerate(data_dict.items()):
        with cols[i]:
            st.markdown(f"**{label}**")
            if not data.empty:
                st.dataframe(data, use_container_width=True, hide_index=True)
            else:
                st.info("No data")

def highlight_critical_rows(df: pd.DataFrame, condition_column: str, threshold: Any) -> pd.DataFrame:
    """
    Highlight rows in a DataFrame based on a condition.
    
    Args:
        df: DataFrame to style
        condition_column: Column to check condition on
        threshold: Threshold value for highlighting
        
    Returns:
        Styled DataFrame
    """
    def highlight_row(row):
        if condition_column in row and row[condition_column] > threshold:
            return ['background-color: #ffebee'] * len(row)
        return [''] * len(row)
    
    return df.style.apply(highlight_row, axis=1) 