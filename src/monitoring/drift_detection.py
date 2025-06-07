"""
Drift detection algorithms for RiskFlow Credit Risk MLOps Pipeline.

Monitors model performance degradation and data distribution shifts.
Provides statistical tests for model drift detection and alerting.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from sqlalchemy import text

from utils.helpers import get_utc_now
from utils.database import DatabaseManager


@dataclass
class DriftResult:
    """Container for drift detection results."""
    
    drift_detected: bool
    drift_score: float
    p_value: Optional[float] = None
    threshold: float = 0.05
    test_statistic: Optional[float] = None
    reference_period: Optional[Tuple[datetime, datetime]] = None
    detection_period: Optional[Tuple[datetime, datetime]] = None
    drift_type: str = "unknown"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DriftDetector(ABC):
    """Abstract base class for drift detection algorithms."""
    
    @abstractmethod
    def detect_drift(self, reference_data: np.ndarray, current_data: np.ndarray) -> DriftResult:
        """Detect drift between reference and current data."""
        pass
    
    @abstractmethod
    def get_drift_score(self, reference_data: np.ndarray, current_data: np.ndarray) -> float:
        """Calculate drift score between datasets."""
        pass


class StatisticalDriftDetector(DriftDetector):
    """
    Statistical drift detection using multiple statistical tests.
    
    Implements various statistical tests for detecting distribution shifts:
    - Kolmogorov-Smirnov test for continuous features
    - Chi-square test for categorical features  
    - Population Stability Index (PSI) for model scores
    - Performance-based drift detection
    """
    
    def __init__(self, 
                 significance_level: float = 0.05,
                 min_sample_size: int = 100,
                 psi_threshold: float = 0.1):
        self.significance_level = significance_level
        self.min_sample_size = min_sample_size
        self.psi_threshold = psi_threshold
    
    def detect_drift(self, reference_data: np.ndarray, current_data: np.ndarray) -> DriftResult:
        """
        Detect drift using statistical tests.
        
        Args:
            reference_data: Reference/baseline data
            current_data: Current data to compare against reference
            
        Returns:
            DriftResult with drift detection results
        """
        if len(reference_data) < self.min_sample_size or len(current_data) < self.min_sample_size:
            return DriftResult(
                drift_detected=False,
                drift_score=0.0,
                drift_type="insufficient_data",
                metadata={"message": "Insufficient data for drift detection"}
            )
        
        # Choose appropriate test based on data characteristics
        if self._is_continuous_data(reference_data):
            return self._ks_test_drift(reference_data, current_data)
        else:
            return self._chi_square_drift(reference_data, current_data)
    
    def get_drift_score(self, reference_data: np.ndarray, current_data: np.ndarray) -> float:
        """Calculate drift score using Population Stability Index (PSI)."""
        return self._calculate_psi(reference_data, current_data)
    
    def _is_continuous_data(self, data: np.ndarray) -> bool:
        """Check if data is continuous or categorical."""
        unique_values = len(np.unique(data))
        return unique_values > 20 or data.dtype in [np.float32, np.float64]
    
    def _ks_test_drift(self, reference_data: np.ndarray, current_data: np.ndarray) -> DriftResult:
        """Perform Kolmogorov-Smirnov test for continuous data."""
        try:
            statistic, p_value = stats.ks_2samp(reference_data, current_data)
            
            drift_detected = p_value < self.significance_level
            
            return DriftResult(
                drift_detected=drift_detected,
                drift_score=statistic,
                p_value=p_value,
                threshold=self.significance_level,
                test_statistic=statistic,
                drift_type="ks_test",
                metadata={
                    "test_name": "Kolmogorov-Smirnov",
                    "reference_size": len(reference_data),
                    "current_size": len(current_data),
                    "reference_mean": np.mean(reference_data),
                    "current_mean": np.mean(current_data),
                    "reference_std": np.std(reference_data),
                    "current_std": np.std(current_data)
                }
            )
            
        except Exception as e:
            return DriftResult(
                drift_detected=False,
                drift_score=0.0,
                drift_type="error",
                metadata={"error": str(e)}
            )
    
    def _chi_square_drift(self, reference_data: np.ndarray, current_data: np.ndarray) -> DriftResult:
        """Perform Chi-square test for categorical data."""
        try:
            # Create frequency tables
            ref_unique, ref_counts = np.unique(reference_data, return_counts=True)
            cur_unique, cur_counts = np.unique(current_data, return_counts=True)
            
            # Align categories
            all_categories = np.union1d(ref_unique, cur_unique)
            
            ref_freq = np.zeros(len(all_categories))
            cur_freq = np.zeros(len(all_categories))
            
            for i, cat in enumerate(all_categories):
                ref_idx = np.where(ref_unique == cat)[0]
                cur_idx = np.where(cur_unique == cat)[0]
                
                if len(ref_idx) > 0:
                    ref_freq[i] = ref_counts[ref_idx[0]]
                if len(cur_idx) > 0:
                    cur_freq[i] = cur_counts[cur_idx[0]]
            
            # Normalize to probabilities
            ref_prob = ref_freq / np.sum(ref_freq)
            cur_prob = cur_freq / np.sum(cur_freq)
            
            # Chi-square test
            expected = ref_prob * np.sum(cur_freq)
            # Add small constant to avoid division by zero
            expected = np.maximum(expected, 1e-10)
            
            chi2_stat = np.sum((cur_freq - expected) ** 2 / expected)
            degrees_of_freedom = len(all_categories) - 1
            p_value = 1 - stats.chi2.cdf(chi2_stat, degrees_of_freedom)
            
            drift_detected = p_value < self.significance_level
            
            return DriftResult(
                drift_detected=drift_detected,
                drift_score=chi2_stat,
                p_value=p_value,
                threshold=self.significance_level,
                test_statistic=chi2_stat,
                drift_type="chi_square",
                metadata={
                    "test_name": "Chi-square",
                    "degrees_of_freedom": degrees_of_freedom,
                    "reference_size": len(reference_data),
                    "current_size": len(current_data),
                    "categories": all_categories.tolist()
                }
            )
            
        except Exception as e:
            return DriftResult(
                drift_detected=False,
                drift_score=0.0,
                drift_type="error",
                metadata={"error": str(e)}
            )
    
    def _calculate_psi(self, reference_data: np.ndarray, current_data: np.ndarray, bins: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        PSI measures the shift in population distribution between two samples.
        PSI < 0.1: No significant change
        0.1 <= PSI < 0.2: Moderate change
        PSI >= 0.2: Significant change
        """
        try:
            # Create bins based on reference data
            if self._is_continuous_data(reference_data):
                bin_edges = np.histogram_bin_edges(reference_data, bins=bins)
                ref_counts, _ = np.histogram(reference_data, bins=bin_edges)
                cur_counts, _ = np.histogram(current_data, bins=bin_edges)
            else:
                # For categorical data, use unique values as bins
                unique_values = np.union1d(np.unique(reference_data), np.unique(current_data))
                ref_counts = np.array([np.sum(reference_data == val) for val in unique_values])
                cur_counts = np.array([np.sum(current_data == val) for val in unique_values])
            
            # Convert to probabilities
            ref_prob = ref_counts / np.sum(ref_counts)
            cur_prob = cur_counts / np.sum(cur_counts)
            
            # Add small constant to avoid log(0)
            ref_prob = np.maximum(ref_prob, 1e-10)
            cur_prob = np.maximum(cur_prob, 1e-10)
            
            # Calculate PSI
            psi = np.sum((cur_prob - ref_prob) * np.log(cur_prob / ref_prob))
            
            return psi
            
        except Exception as e:
            print(f"Error calculating PSI: {e}")
            return 0.0
    
    def detect_performance_drift(self, 
                               reference_predictions: np.ndarray,
                               reference_actuals: np.ndarray,
                               current_predictions: np.ndarray,
                               current_actuals: np.ndarray,
                               metric: str = "accuracy") -> DriftResult:
        """
        Detect performance drift by comparing model performance metrics.
        
        Args:
            reference_predictions: Reference model predictions
            reference_actuals: Reference actual values
            current_predictions: Current model predictions
            current_actuals: Current actual values
            metric: Performance metric to compare ("accuracy", "auc", "precision", "recall")
            
        Returns:
            DriftResult with performance drift detection results
        """
        try:
            # Calculate performance metrics
            if metric == "accuracy":
                ref_performance = accuracy_score(reference_actuals, reference_predictions > 0.5)
                cur_performance = accuracy_score(current_actuals, current_predictions > 0.5)
            elif metric == "auc":
                ref_performance = roc_auc_score(reference_actuals, reference_predictions)
                cur_performance = roc_auc_score(current_actuals, current_predictions)
            elif metric == "precision":
                ref_performance = precision_score(reference_actuals, reference_predictions > 0.5)
                cur_performance = precision_score(current_actuals, current_predictions > 0.5)
            elif metric == "recall":
                ref_performance = recall_score(reference_actuals, reference_predictions > 0.5)
                cur_performance = recall_score(current_actuals, current_predictions > 0.5)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            
            # Calculate performance difference
            performance_diff = abs(ref_performance - cur_performance)
            
            # Use relative threshold (5% degradation)
            relative_threshold = 0.05
            drift_detected = performance_diff > (ref_performance * relative_threshold)
            
            return DriftResult(
                drift_detected=drift_detected,
                drift_score=performance_diff,
                threshold=relative_threshold,
                drift_type="performance_drift",
                metadata={
                    "metric": metric,
                    "reference_performance": ref_performance,
                    "current_performance": cur_performance,
                    "performance_difference": performance_diff,
                    "relative_change": performance_diff / ref_performance if ref_performance > 0 else 0
                }
            )
            
        except Exception as e:
            return DriftResult(
                drift_detected=False,
                drift_score=0.0,
                drift_type="error",
                metadata={"error": str(e), "metric": metric}
            )


class ModelDriftMonitor:
    """
    Comprehensive model drift monitoring system.
    
    Monitors multiple aspects of model drift:
    - Feature drift (input data changes)
    - Prediction drift (model output changes)
    - Performance drift (model accuracy changes)
    """
    
    def __init__(self, 
                 db_manager: Optional[DatabaseManager] = None,
                 drift_detector: Optional[DriftDetector] = None):
        self.db_manager = db_manager or DatabaseManager()
        self.drift_detector = drift_detector or StatisticalDriftDetector()
        
        # Initialize drift monitoring tables
        self._init_drift_tables()
    
    def _init_drift_tables(self):
        """Initialize drift monitoring tables in database."""
        try:
            with self.db_manager.get_session() as session:
                session.execute(text("""
                    CREATE TABLE IF NOT EXISTS drift_detections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME NOT NULL,
                        model_name TEXT NOT NULL,
                        model_version TEXT NOT NULL,
                        drift_type TEXT NOT NULL,
                        drift_detected BOOLEAN NOT NULL,
                        drift_score REAL NOT NULL,
                        p_value REAL,
                        threshold_value REAL,
                        test_statistic REAL,
                        reference_start DATETIME,
                        reference_end DATETIME,
                        detection_start DATETIME,
                        detection_end DATETIME,
                        metadata TEXT
                    )
                """))
                session.commit()
                
        except Exception as e:
            print(f"Error initializing drift tables: {e}")
    
    def monitor_feature_drift(self, 
                            model_name: str,
                            model_version: str,
                            feature_name: str,
                            reference_data: np.ndarray,
                            current_data: np.ndarray) -> DriftResult:
        """Monitor drift in a specific feature."""
        
        result = self.drift_detector.detect_drift(reference_data, current_data)
        result.drift_type = f"feature_drift_{feature_name}"
        
        # Store result in database
        self._store_drift_result(model_name, model_version, result)
        
        return result
    
    def monitor_prediction_drift(self,
                               model_name: str,
                               model_version: str,
                               reference_predictions: np.ndarray,
                               current_predictions: np.ndarray) -> DriftResult:
        """Monitor drift in model predictions."""
        
        result = self.drift_detector.detect_drift(reference_predictions, current_predictions)
        result.drift_type = "prediction_drift"
        
        # Store result in database
        self._store_drift_result(model_name, model_version, result)
        
        return result
    
    def monitor_performance_drift(self,
                                model_name: str,
                                model_version: str,
                                reference_predictions: np.ndarray,
                                reference_actuals: np.ndarray,
                                current_predictions: np.ndarray,
                                current_actuals: np.ndarray) -> DriftResult:
        """Monitor drift in model performance."""
        
        if isinstance(self.drift_detector, StatisticalDriftDetector):
            result = self.drift_detector.detect_performance_drift(
                reference_predictions, reference_actuals,
                current_predictions, current_actuals
            )
        else:
            # Fallback to prediction drift
            result = self.drift_detector.detect_drift(reference_predictions, current_predictions)
            result.drift_type = "performance_drift"
        
        # Store result in database
        self._store_drift_result(model_name, model_version, result)
        
        return result
    
    def _store_drift_result(self, model_name: str, model_version: str, result: DriftResult):
        """Store drift detection result in database."""
        try:
            with self.db_manager.get_session() as session:
                session.execute(text("""
                    INSERT INTO drift_detections (
                        timestamp, model_name, model_version, drift_type, drift_detected,
                        drift_score, p_value, threshold_value, test_statistic,
                        reference_start, reference_end, detection_start, detection_end, metadata
                    ) VALUES (
                        :timestamp, :model_name, :model_version, :drift_type, :drift_detected,
                        :drift_score, :p_value, :threshold_value, :test_statistic,
                        :reference_start, :reference_end, :detection_start, :detection_end, :metadata
                    )
                """), {
                    'timestamp': get_utc_now(),
                    'model_name': model_name,
                    'model_version': model_version,
                    'drift_type': result.drift_type,
                    'drift_detected': result.drift_detected,
                    'drift_score': result.drift_score,
                    'p_value': result.p_value,
                    'threshold_value': result.threshold,
                    'test_statistic': result.test_statistic,
                    'reference_start': result.reference_period[0] if result.reference_period else None,
                    'reference_end': result.reference_period[1] if result.reference_period else None,
                    'detection_start': result.detection_period[0] if result.detection_period else None,
                    'detection_end': result.detection_period[1] if result.detection_period else None,
                    'metadata': str(result.metadata) if result.metadata else None
                })
                session.commit()
                
        except Exception as e:
            print(f"Error storing drift result: {e}")
    
    def get_drift_history(self, 
                         model_name: Optional[str] = None,
                         drift_type: Optional[str] = None,
                         hours: int = 24) -> List[Dict[str, Any]]:
        """Get drift detection history from database."""
        try:
            with self.db_manager.get_session() as session:
                query = """
                    SELECT * FROM drift_detections 
                    WHERE timestamp >= :start_time
                """
                params = {
                    'start_time': get_utc_now() - timedelta(hours=hours)
                }
                
                if model_name:
                    query += " AND model_name = :model_name"
                    params['model_name'] = model_name
                
                if drift_type:
                    query += " AND drift_type = :drift_type"
                    params['drift_type'] = drift_type
                
                query += " ORDER BY timestamp DESC"
                
                result = session.execute(text(query), params)
                return [dict(row._mapping) for row in result]
                
        except Exception as e:
            print(f"Error getting drift history: {e}")
            return [] 