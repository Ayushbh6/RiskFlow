"""
Monitoring and observability module for RiskFlow Credit Risk MLOps Pipeline.

This module provides:
- Model performance tracking
- API metrics collection  
- Drift detection algorithms
- Alerting system
"""

from .dashboard_data import DashboardDataProvider

__all__ = [
    'DashboardDataProvider'
] 