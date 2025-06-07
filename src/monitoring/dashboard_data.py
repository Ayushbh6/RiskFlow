"""
Dashboard data provider for RiskFlow Credit Risk MLOps Pipeline.

Provides aggregated data and metrics for Streamlit dashboard visualization.
Optimized for real-time dashboard updates with caching and efficient queries.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import threading
from collections import defaultdict
from sqlalchemy import text

from config.settings import get_settings
from utils.database import DatabaseManager
from utils.helpers import get_utc_now
from monitoring.metrics import MetricsCollector
from monitoring.drift_detection import ModelDriftMonitor
from monitoring.alerting import AlertManager


@dataclass
class DashboardMetrics:
    """Container for dashboard metrics."""
    
    # API Metrics
    total_requests: int = 0
    avg_response_time: float = 0.0
    error_rate: float = 0.0
    requests_per_minute: float = 0.0
    
    # Model Metrics
    total_predictions: int = 0
    avg_accuracy: float = 0.0
    high_risk_rate: float = 0.0
    avg_prediction_time: float = 0.0
    
    # System Metrics
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    
    # Alert Metrics
    active_alerts: int = 0
    critical_alerts: int = 0
    
    # Drift Metrics
    feature_drift_count: int = 0
    performance_drift_detected: bool = False
    
    # Timestamp
    last_updated: datetime = None


class DashboardDataProvider:
    """
    Comprehensive data provider for RiskFlow dashboard.
    
    Aggregates data from multiple sources:
    - Metrics collector
    - Alert manager
    - Drift monitor
    - Database queries
    
    Provides cached, real-time data for dashboard visualization.
    """
    
    def __init__(self, 
                 db_manager: Optional[DatabaseManager] = None,
                 metrics_collector: Optional[MetricsCollector] = None,
                 alert_manager: Optional[AlertManager] = None,
                 drift_monitor: Optional[ModelDriftMonitor] = None):
        
        self.settings = get_settings()
        self.db_manager = db_manager or DatabaseManager()
        self.metrics_collector = metrics_collector or MetricsCollector(self.db_manager)
        self.alert_manager = alert_manager or AlertManager(self.db_manager)
        self.drift_monitor = drift_monitor or ModelDriftMonitor(self.db_manager)
        
        # Cache settings
        self._cache_ttl = 30  # seconds
        self._cache = {}
        self._cache_timestamps = {}
        self._lock = threading.RLock()
    
    def get_dashboard_summary(self, refresh_cache: bool = False) -> DashboardMetrics:
        """Get high-level dashboard metrics summary."""
        
        cache_key = "dashboard_summary"
        
        if not refresh_cache and self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        with self._lock:
            try:
                # Get metrics from collectors
                metrics_summary = self.metrics_collector.get_metrics_summary()
                active_alerts = self.alert_manager.get_active_alerts()
                
                # Get system metrics
                system_metrics = self._get_latest_system_metrics()
                
                # Get drift information
                drift_info = self._get_drift_summary()
                
                # Create dashboard metrics
                dashboard_metrics = DashboardMetrics(
                    # API metrics
                    total_requests=metrics_summary.get('api_metrics', {}).get('total_requests', 0),
                    avg_response_time=metrics_summary.get('api_metrics', {}).get('avg_response_time', 0.0),
                    error_rate=metrics_summary.get('api_metrics', {}).get('error_rate', 0.0),
                    requests_per_minute=metrics_summary.get('api_metrics', {}).get('requests_per_minute', 0.0),
                    
                    # Model metrics
                    total_predictions=metrics_summary.get('model_metrics', {}).get('total_predictions', 0),
                    avg_accuracy=metrics_summary.get('model_metrics', {}).get('avg_accuracy', 0.0),
                    high_risk_rate=metrics_summary.get('model_metrics', {}).get('high_risk_rate', 0.0),
                    avg_prediction_time=metrics_summary.get('model_metrics', {}).get('avg_prediction_time', 0.0),
                    
                    # System metrics
                    cpu_usage=system_metrics.get('cpu_percent', 0.0),
                    memory_usage=system_metrics.get('memory_percent', 0.0),
                    disk_usage=system_metrics.get('disk_percent', 0.0),
                    
                    # Alert metrics
                    active_alerts=len(active_alerts),
                    critical_alerts=len([a for a in active_alerts if a.severity.value == 'critical']),
                    
                    # Drift metrics
                    feature_drift_count=drift_info.get('feature_drift_count', 0),
                    performance_drift_detected=drift_info.get('performance_drift_detected', False),
                    
                    last_updated=get_utc_now()
                )
                
                # Cache result
                self._cache[cache_key] = dashboard_metrics
                self._cache_timestamps[cache_key] = get_utc_now()
                
                return dashboard_metrics
                
            except Exception as e:
                print(f"Error getting dashboard summary: {e}")
                return DashboardMetrics(last_updated=get_utc_now())
    
    def get_api_metrics_timeseries(self, hours: int = 24) -> pd.DataFrame:
        """Get API metrics time series data."""
        
        cache_key = f"api_timeseries_{hours}"
        
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        try:
            api_metrics = self.metrics_collector.get_api_metrics_history(hours=hours)
            
            if not api_metrics:
                return pd.DataFrame()
            
            df = pd.DataFrame(api_metrics)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Resample to 5-minute intervals for visualization
            df.set_index('timestamp', inplace=True)
            
            # Aggregate metrics
            agg_df = df.resample('5T').agg({
                'response_time': ['mean', 'p95'],
                'status_code': lambda x: (x >= 400).sum() / len(x) if len(x) > 0 else 0,  # Error rate
                'endpoint': 'count'  # Request count
            }).reset_index()
            
            # Flatten column names
            agg_df.columns = ['timestamp', 'avg_response_time', 'p95_response_time', 'error_rate', 'request_count']
            
            # Cache result
            self._cache[cache_key] = agg_df
            self._cache_timestamps[cache_key] = get_utc_now()
            
            return agg_df
            
        except Exception as e:
            print(f"Error getting API metrics timeseries: {e}")
            return pd.DataFrame()
    
    def get_model_performance_timeseries(self, model_name: Optional[str] = None, hours: int = 24) -> pd.DataFrame:
        """Get model performance time series data."""
        
        cache_key = f"model_timeseries_{model_name}_{hours}"
        
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        try:
            model_metrics = self.metrics_collector.get_model_metrics_history(model_name=model_name, hours=hours)
            
            if not model_metrics:
                return pd.DataFrame()
            
            df = pd.DataFrame(model_metrics)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Cache result
            self._cache[cache_key] = df
            self._cache_timestamps[cache_key] = get_utc_now()
            
            return df
            
        except Exception as e:
            print(f"Error getting model performance timeseries: {e}")
            return pd.DataFrame()
    
    def get_system_metrics_timeseries(self, hours: int = 24) -> pd.DataFrame:
        """Get system metrics time series data."""
        
        cache_key = f"system_timeseries_{hours}"
        
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        try:
            system_metrics = self.metrics_collector.get_system_metrics_history(hours=hours)
            
            if not system_metrics:
                return pd.DataFrame()
            
            df = pd.DataFrame(system_metrics)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Cache result
            self._cache[cache_key] = df
            self._cache_timestamps[cache_key] = get_utc_now()
            
            return df
            
        except Exception as e:
            print(f"Error getting system metrics timeseries: {e}")
            return pd.DataFrame()
    
    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert summary and recent alerts."""
        
        cache_key = f"alert_summary_{hours}"
        
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        try:
            # Get active alerts
            active_alerts = self.alert_manager.get_active_alerts()
            
            # Get alert history
            alert_history = self.alert_manager.get_alert_history(hours=hours)
            
            # Aggregate by severity
            severity_counts = defaultdict(int)
            for alert in active_alerts:
                severity_counts[alert.severity.value] += 1
            
            # Recent alerts trend
            df = pd.DataFrame(alert_history) if alert_history else pd.DataFrame()
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                
                # Hourly alert counts
                hourly_counts = df.resample('1H').size().reset_index()
                hourly_counts.columns = ['timestamp', 'alert_count']
            else:
                hourly_counts = pd.DataFrame(columns=['timestamp', 'alert_count'])
            
            summary = {
                'active_alerts': len(active_alerts),
                'severity_counts': dict(severity_counts),
                'total_alerts_24h': len(alert_history),
                'hourly_trend': hourly_counts.to_dict('records'),
                'recent_alerts': alert_history[:10] if alert_history else []
            }
            
            # Cache result
            self._cache[cache_key] = summary
            self._cache_timestamps[cache_key] = get_utc_now()
            
            return summary
            
        except Exception as e:
            print(f"Error getting alert summary: {e}")
            return {
                'active_alerts': 0,
                'severity_counts': {},
                'total_alerts_24h': 0,
                'hourly_trend': [],
                'recent_alerts': []
            }
    
    def get_drift_analysis(self, hours: int = 24) -> Dict[str, Any]:
        """Get drift detection analysis."""
        
        cache_key = f"drift_analysis_{hours}"
        
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        try:
            drift_history = self.drift_monitor.get_drift_history(hours=hours)
            
            if not drift_history:
                return {
                    'total_drift_detections': 0,
                    'feature_drift_count': 0,
                    'performance_drift_count': 0,
                    'drift_trend': [],
                    'recent_drift_events': []
                }
            
            df = pd.DataFrame(drift_history)
            
            # Count drift types
            feature_drift = df[df['drift_type'].str.contains('feature_drift', na=False)]
            performance_drift = df[df['drift_type'] == 'performance_drift']
            
            # Drift trend over time
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            hourly_drift = df.groupby([df.index.floor('H'), 'drift_type']).size().reset_index()
            hourly_drift.columns = ['timestamp', 'drift_type', 'count']
            
            analysis = {
                'total_drift_detections': len(drift_history),
                'feature_drift_count': len(feature_drift),
                'performance_drift_count': len(performance_drift),
                'drift_trend': hourly_drift.to_dict('records'),
                'recent_drift_events': drift_history[:10]
            }
            
            # Cache result
            self._cache[cache_key] = analysis
            self._cache_timestamps[cache_key] = get_utc_now()
            
            return analysis
            
        except Exception as e:
            print(f"Error getting drift analysis: {e}")
            return {
                'total_drift_detections': 0,
                'feature_drift_count': 0,
                'performance_drift_count': 0,
                'drift_trend': [],
                'recent_drift_events': []
            }
    
    def get_prediction_analysis(self, hours: int = 24) -> Dict[str, Any]:
        """Get prediction analysis and statistics."""
        
        cache_key = f"prediction_analysis_{hours}"
        
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        try:
            # Get prediction logs from database
            with self.db_manager.get_session() as session:
                query = """
                    SELECT * FROM prediction_log 
                    WHERE timestamp >= :start_time
                    ORDER BY timestamp DESC
                """
                params = {
                    'start_time': get_utc_now() - timedelta(hours=hours)
                }
                
                result = session.execute(text(query), params)
                predictions = [dict(row._mapping) for row in result]
            
            if not predictions:
                return {
                    'total_predictions': 0,
                    'avg_pd_score': 0.0,
                    'avg_lgd_score': 0.0,
                    'high_risk_rate': 0.0,
                    'prediction_distribution': [],
                    'hourly_prediction_count': []
                }
            
            df = pd.DataFrame(predictions)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Parse features if stored as JSON
            if 'features' in df.columns and not df['features'].empty:
                features_data = df['features'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
                pd_scores = features_data.apply(lambda x: x.get('pd_score', 0.0) if isinstance(x, dict) else 0.0)
                lgd_scores = features_data.apply(lambda x: x.get('lgd_score', 0.0) if isinstance(x, dict) else 0.0)
            else:
                pd_scores = df.get('pd_score', pd.Series([0.0] * len(df)))
                lgd_scores = df.get('lgd_score', pd.Series([0.0] * len(df)))
            
            # High risk threshold (e.g., PD > 0.1 or 10%)
            high_risk_threshold = 0.1
            high_risk_count = (pd_scores > high_risk_threshold).sum()
            
            # Hourly prediction counts
            df.set_index('timestamp', inplace=True)
            hourly_counts = df.resample('1H').size().reset_index()
            hourly_counts.columns = ['timestamp', 'prediction_count']
            
            # PD score distribution (bins)
            pd_bins = np.histogram(pd_scores, bins=10, range=(0, 1))
            pd_distribution = [
                {'bin_start': pd_bins[1][i], 'bin_end': pd_bins[1][i+1], 'count': int(pd_bins[0][i])}
                for i in range(len(pd_bins[0]))
            ]
            
            analysis = {
                'total_predictions': len(predictions),
                'avg_pd_score': float(pd_scores.mean()) if len(pd_scores) > 0 else 0.0,
                'avg_lgd_score': float(lgd_scores.mean()) if len(lgd_scores) > 0 else 0.0,
                'high_risk_rate': float(high_risk_count / len(predictions)) if len(predictions) > 0 else 0.0,
                'prediction_distribution': pd_distribution,
                'hourly_prediction_count': hourly_counts.to_dict('records')
            }
            
            # Cache result
            self._cache[cache_key] = analysis
            self._cache_timestamps[cache_key] = get_utc_now()
            
            return analysis
            
        except Exception as e:
            print(f"Error getting prediction analysis: {e}")
            return {
                'total_predictions': 0,
                'avg_pd_score': 0.0,
                'avg_lgd_score': 0.0,
                'high_risk_rate': 0.0,
                'prediction_distribution': [],
                'hourly_prediction_count': []
            }
    
    def get_model_registry_info(self) -> List[Dict[str, Any]]:
        """Get model registry information."""
        
        cache_key = "model_registry_info"
        
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        try:
            with self.db_manager.get_session() as session:
                query = """
                    SELECT * FROM model_registry 
                    ORDER BY created_at DESC
                """
                
                result = session.execute(text(query))
                models = [dict(row._mapping) for row in result]
            
            # Cache result
            self._cache[cache_key] = models
            self._cache_timestamps[cache_key] = get_utc_now()
            
            return models
            
        except Exception as e:
            print(f"Error getting model registry info: {e}")
            return []
    
    def _get_latest_system_metrics(self) -> Dict[str, Any]:
        """Get latest system metrics."""
        try:
            system_metrics = self.metrics_collector.get_system_metrics_history(hours=1)
            if system_metrics:
                return system_metrics[0]  # Most recent
            return {}
        except Exception as e:
            print(f"Error getting latest system metrics: {e}")
            return {}
    
    def _get_drift_summary(self) -> Dict[str, Any]:
        """Get drift detection summary."""
        try:
            drift_history = self.drift_monitor.get_drift_history(hours=24)
            
            feature_drift_count = len([d for d in drift_history if 'feature_drift' in d.get('drift_type', '')])
            performance_drift_detected = any(d.get('drift_type') == 'performance_drift' and d.get('drift_detected', False) for d in drift_history)
            
            return {
                'feature_drift_count': feature_drift_count,
                'performance_drift_detected': performance_drift_detected
            }
        except Exception as e:
            print(f"Error getting drift summary: {e}")
            return {'feature_drift_count': 0, 'performance_drift_detected': False}
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self._cache or cache_key not in self._cache_timestamps:
            return False
        
        cache_age = (get_utc_now() - self._cache_timestamps[cache_key]).total_seconds()
        return cache_age < self._cache_ttl
    
    def clear_cache(self):
        """Clear all cached data."""
        with self._lock:
            self._cache.clear()
            self._cache_timestamps.clear()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status."""
        
        try:
            # Get latest metrics
            dashboard_metrics = self.get_dashboard_summary()
            
            # Determine health status
            health_score = 100
            issues = []
            
            # Check API health
            if dashboard_metrics.error_rate > 0.05:  # 5% error rate
                health_score -= 20
                issues.append(f"High API error rate: {dashboard_metrics.error_rate:.2%}")
            
            if dashboard_metrics.avg_response_time > 1.0:  # 1 second
                health_score -= 15
                issues.append(f"Slow API response time: {dashboard_metrics.avg_response_time:.2f}s")
            
            # Check system resources
            if dashboard_metrics.memory_usage > 90:
                health_score -= 25
                issues.append(f"High memory usage: {dashboard_metrics.memory_usage:.1f}%")
            
            if dashboard_metrics.cpu_usage > 80:
                health_score -= 20
                issues.append(f"High CPU usage: {dashboard_metrics.cpu_usage:.1f}%")
            
            # Check alerts
            if dashboard_metrics.critical_alerts > 0:
                health_score -= 30
                issues.append(f"Critical alerts active: {dashboard_metrics.critical_alerts}")
            
            # Check drift
            if dashboard_metrics.performance_drift_detected:
                health_score -= 25
                issues.append("Performance drift detected")
            
            # Determine status
            if health_score >= 90:
                status = "healthy"
                color = "green"
            elif health_score >= 70:
                status = "warning"
                color = "yellow"
            else:
                status = "critical"
                color = "red"
            
            return {
                'status': status,
                'health_score': max(0, health_score),
                'color': color,
                'issues': issues,
                'last_updated': get_utc_now().isoformat()
            }
            
        except Exception as e:
            print(f"Error getting health status: {e}")
            return {
                'status': 'unknown',
                'health_score': 0,
                'color': 'gray',
                'issues': [f"Health check error: {str(e)}"],
                'last_updated': get_utc_now().isoformat()
            } 