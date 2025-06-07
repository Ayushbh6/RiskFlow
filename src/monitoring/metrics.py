"""
Metrics collection and tracking for RiskFlow Credit Risk MLOps Pipeline.

Provides comprehensive metrics collection for:
- Model performance tracking
- API request/response monitoring
- System resource utilization
- Business KPIs
"""

import time
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
import json
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from config.settings import get_settings
from utils.database import DatabaseManager
from utils.helpers import get_utc_now


@dataclass
class ModelMetrics:
    """Model performance metrics container."""
    
    model_name: str
    model_version: str
    timestamp: datetime
    
    # Performance metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_score: Optional[float] = None
    
    # Prediction metrics
    prediction_count: int = 0
    avg_prediction_time: float = 0.0
    
    # Business metrics
    avg_pd_score: Optional[float] = None
    avg_lgd_score: Optional[float] = None
    high_risk_predictions: int = 0
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class APIMetrics:
    """API performance metrics container."""
    
    endpoint: str
    method: str
    timestamp: datetime
    
    # Performance metrics
    response_time: float
    status_code: int
    request_size: int = 0
    response_size: int = 0
    
    # Error tracking
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    
    # Client information
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """
    Comprehensive metrics collection system for RiskFlow.
    
    Tracks model performance, API metrics, and system health.
    Thread-safe with in-memory buffering and database persistence.
    """
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.settings = get_settings()
        self.db_manager = db_manager or DatabaseManager()
        
        # In-memory metric storage
        self._model_metrics: deque = deque(maxlen=1000)
        self._api_metrics: deque = deque(maxlen=10000)
        self._system_metrics: deque = deque(maxlen=100)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Aggregated metrics cache
        self._metrics_cache: Dict[str, Any] = {}
        self._last_cache_update = get_utc_now()
        
        # Initialize database tables
        self._init_metrics_tables()
    
    def _init_metrics_tables(self):
        """Initialize metrics tables in database."""
        try:
            with self.db_manager.get_session() as session:
                # Create metrics tables if they don't exist
                session.execute(text("""
                    CREATE TABLE IF NOT EXISTS model_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_name TEXT NOT NULL,
                        model_version TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        accuracy REAL,
                        precision_score REAL,
                        recall_score REAL,
                        f1_score REAL,
                        auc_score REAL,
                        prediction_count INTEGER DEFAULT 0,
                        avg_prediction_time REAL DEFAULT 0.0,
                        avg_pd_score REAL,
                        avg_lgd_score REAL,
                        high_risk_predictions INTEGER DEFAULT 0,
                        metadata TEXT
                    )
                """))
                
                session.execute(text("""
                    CREATE TABLE IF NOT EXISTS api_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        endpoint TEXT NOT NULL,
                        method TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        response_time REAL NOT NULL,
                        status_code INTEGER NOT NULL,
                        request_size INTEGER DEFAULT 0,
                        response_size INTEGER DEFAULT 0,
                        error_type TEXT,
                        error_message TEXT,
                        client_ip TEXT,
                        user_agent TEXT,
                        metadata TEXT
                    )
                """))
                
                session.execute(text("""
                    CREATE TABLE IF NOT EXISTS system_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME NOT NULL,
                        cpu_percent REAL,
                        memory_percent REAL,
                        disk_percent REAL,
                        active_connections INTEGER DEFAULT 0,
                        total_requests INTEGER DEFAULT 0,
                        error_rate REAL DEFAULT 0.0,
                        metadata TEXT
                    )
                """))
                
                session.commit()
                
        except Exception as e:
            print(f"Error initializing metrics tables: {e}")
    
    def record_model_metrics(self, metrics: ModelMetrics):
        """Record model performance metrics."""
        with self._lock:
            # Add to in-memory storage
            self._model_metrics.append(metrics)
            
            # Persist to database
            self._persist_model_metrics(metrics)
            
            # Update cache
            self._update_metrics_cache()
    
    def record_api_metrics(self, metrics: APIMetrics):
        """Record API performance metrics."""
        with self._lock:
            # Add to in-memory storage
            self._api_metrics.append(metrics)
            
            # Persist to database
            self._persist_api_metrics(metrics)
            
            # Update cache
            self._update_metrics_cache()
    
    def record_system_metrics(self):
        """Record current system metrics."""
        try:
            # Collect system metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Get API metrics for error rate calculation
            recent_api_metrics = list(self._api_metrics)[-100:]  # Last 100 requests
            total_requests = len(recent_api_metrics)
            error_count = sum(1 for m in recent_api_metrics if m.status_code >= 400)
            error_rate = error_count / total_requests if total_requests > 0 else 0.0
            
            system_metrics = {
                'timestamp': get_utc_now(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': disk.percent,
                'active_connections': 0,  # Would need specific monitoring
                'total_requests': total_requests,
                'error_rate': error_rate,
                'metadata': json.dumps({
                    'memory_available': memory.available,
                    'disk_free': disk.free,
                    'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
                })
            }
            
            with self._lock:
                self._system_metrics.append(system_metrics)
                self._persist_system_metrics(system_metrics)
                
        except Exception as e:
            print(f"Error recording system metrics: {e}")
    
    def _persist_model_metrics(self, metrics: ModelMetrics):
        """Persist model metrics to database."""
        try:
            with self.db_manager.get_session() as session:
                session.execute(text("""
                    INSERT INTO model_metrics (
                        model_name, model_version, timestamp, accuracy, precision_score,
                        recall_score, f1_score, auc_score, prediction_count, avg_prediction_time,
                        avg_pd_score, avg_lgd_score, high_risk_predictions, metadata
                    ) VALUES (
                        :model_name, :model_version, :timestamp, :accuracy, :precision_score,
                        :recall_score, :f1_score, :auc_score, :prediction_count, :avg_prediction_time,
                        :avg_pd_score, :avg_lgd_score, :high_risk_predictions, :metadata
                    )
                """), {
                    'model_name': metrics.model_name,
                    'model_version': metrics.model_version,
                    'timestamp': metrics.timestamp,
                    'accuracy': metrics.accuracy,
                    'precision_score': metrics.precision,
                    'recall_score': metrics.recall,
                    'f1_score': metrics.f1_score,
                    'auc_score': metrics.auc_score,
                    'prediction_count': metrics.prediction_count,
                    'avg_prediction_time': metrics.avg_prediction_time,
                    'avg_pd_score': metrics.avg_pd_score,
                    'avg_lgd_score': metrics.avg_lgd_score,
                    'high_risk_predictions': metrics.high_risk_predictions,
                    'metadata': json.dumps(metrics.metadata)
                })
                session.commit()
                
        except Exception as e:
            print(f"Error persisting model metrics: {e}")
    
    def _persist_api_metrics(self, metrics: APIMetrics):
        """Persist API metrics to database."""
        try:
            with self.db_manager.get_session() as session:
                session.execute(text("""
                    INSERT INTO api_metrics (
                        endpoint, method, timestamp, response_time, status_code,
                        request_size, response_size, error_type, error_message,
                        client_ip, user_agent, metadata
                    ) VALUES (
                        :endpoint, :method, :timestamp, :response_time, :status_code,
                        :request_size, :response_size, :error_type, :error_message,
                        :client_ip, :user_agent, :metadata
                    )
                """), {
                    'endpoint': metrics.endpoint,
                    'method': metrics.method,
                    'timestamp': metrics.timestamp,
                    'response_time': metrics.response_time,
                    'status_code': metrics.status_code,
                    'request_size': metrics.request_size,
                    'response_size': metrics.response_size,
                    'error_type': metrics.error_type,
                    'error_message': metrics.error_message,
                    'client_ip': metrics.client_ip,
                    'user_agent': metrics.user_agent,
                    'metadata': json.dumps(metrics.metadata)
                })
                session.commit()
                
        except Exception as e:
            print(f"Error persisting API metrics: {e}")
    
    def _persist_system_metrics(self, metrics: Dict[str, Any]):
        """Persist system metrics to database."""
        try:
            with self.db_manager.get_session() as session:
                session.execute(text("""
                    INSERT INTO system_metrics (
                        timestamp, cpu_percent, memory_percent, disk_percent,
                        active_connections, total_requests, error_rate, metadata
                    ) VALUES (
                        :timestamp, :cpu_percent, :memory_percent, :disk_percent,
                        :active_connections, :total_requests, :error_rate, :metadata
                    )
                """), metrics)
                session.commit()
                
        except Exception as e:
            print(f"Error persisting system metrics: {e}")
    
    def _update_metrics_cache(self):
        """Update aggregated metrics cache."""
        try:
            now = get_utc_now()
            
            # Update cache every 30 seconds
            if (now - self._last_cache_update).total_seconds() < 30:
                return
            
            # Calculate aggregated metrics
            recent_api_metrics = [m for m in self._api_metrics if (now - m.timestamp).total_seconds() < 300]  # Last 5 minutes
            recent_model_metrics = [m for m in self._model_metrics if (now - m.timestamp).total_seconds() < 3600]  # Last hour
            
            # API metrics aggregation
            if recent_api_metrics:
                response_times = [m.response_time for m in recent_api_metrics]
                status_codes = [m.status_code for m in recent_api_metrics]
                
                api_summary = {
                    'total_requests': len(recent_api_metrics),
                    'avg_response_time': np.mean(response_times),
                    'p95_response_time': np.percentile(response_times, 95),
                    'p99_response_time': np.percentile(response_times, 99),
                    'error_rate': sum(1 for code in status_codes if code >= 400) / len(status_codes),
                    'requests_per_minute': len(recent_api_metrics) / 5.0
                }
            else:
                api_summary = {
                    'total_requests': 0,
                    'avg_response_time': 0.0,
                    'p95_response_time': 0.0,
                    'p99_response_time': 0.0,
                    'error_rate': 0.0,
                    'requests_per_minute': 0.0
                }
            
            # Model metrics aggregation
            if recent_model_metrics:
                model_summary = {
                    'total_predictions': sum(m.prediction_count for m in recent_model_metrics),
                    'avg_accuracy': np.mean([m.accuracy for m in recent_model_metrics if m.accuracy is not None]),
                    'avg_prediction_time': np.mean([m.avg_prediction_time for m in recent_model_metrics if m.avg_prediction_time > 0]),
                    'high_risk_rate': sum(m.high_risk_predictions for m in recent_model_metrics) / max(sum(m.prediction_count for m in recent_model_metrics), 1)
                }
            else:
                model_summary = {
                    'total_predictions': 0,
                    'avg_accuracy': 0.0,
                    'avg_prediction_time': 0.0,
                    'high_risk_rate': 0.0
                }
            
            # Update cache
            self._metrics_cache = {
                'api_metrics': api_summary,
                'model_metrics': model_summary,
                'last_updated': now
            }
            
            self._last_cache_update = now
            
        except Exception as e:
            print(f"Error updating metrics cache: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get current metrics summary."""
        with self._lock:
            self._update_metrics_cache()
            return self._metrics_cache.copy()
    
    def get_model_metrics_history(self, model_name: Optional[str] = None, hours: int = 24) -> List[Dict[str, Any]]:
        """Get model metrics history from database."""
        try:
            with self.db_manager.get_session() as session:
                query = """
                    SELECT * FROM model_metrics 
                    WHERE timestamp >= :start_time
                """
                params = {
                    'start_time': get_utc_now() - timedelta(hours=hours)
                }
                
                if model_name:
                    query += " AND model_name = :model_name"
                    params['model_name'] = model_name
                
                query += " ORDER BY timestamp DESC"
                
                result = session.execute(text(query), params)
                return [dict(row._mapping) for row in result]
                
        except Exception as e:
            print(f"Error getting model metrics history: {e}")
            return []
    
    def get_api_metrics_history(self, endpoint: Optional[str] = None, hours: int = 24) -> List[Dict[str, Any]]:
        """Get API metrics history from database."""
        try:
            with self.db_manager.get_session() as session:
                query = """
                    SELECT * FROM api_metrics 
                    WHERE timestamp >= :start_time
                """
                params = {
                    'start_time': get_utc_now() - timedelta(hours=hours)
                }
                
                if endpoint:
                    query += " AND endpoint = :endpoint"
                    params['endpoint'] = endpoint
                
                query += " ORDER BY timestamp DESC"
                
                result = session.execute(text(query), params)
                return [dict(row._mapping) for row in result]
                
        except Exception as e:
            print(f"Error getting API metrics history: {e}")
            return []
    
    def get_system_metrics_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get system metrics history from database."""
        try:
            with self.db_manager.get_session() as session:
                query = """
                    SELECT * FROM system_metrics 
                    WHERE timestamp >= :start_time
                    ORDER BY timestamp DESC
                """
                params = {
                    'start_time': get_utc_now() - timedelta(hours=hours)
                }
                
                result = session.execute(text(query), params)
                return [dict(row._mapping) for row in result]
                
        except Exception as e:
            print(f"Error getting system metrics history: {e}")
            return [] 