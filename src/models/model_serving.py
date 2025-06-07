"""
Model serving utilities for RiskFlow Credit Risk MLOps.
Handles model loading, prediction, and serving optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
import time
from pathlib import Path
import hashlib
from functools import lru_cache
import logging
import os
import joblib
from cachetools import TTLCache

from models.credit_risk_model import CreditRiskModel
from models.model_training import ModelTrainingPipeline
from data.preprocessing import CreditRiskFeatureEngineer, get_current_economic_features
from data.validators import CreditDataSchema, validate_real_time_data
from utils.database import db_manager, get_db_session
from config.settings import get_settings
from config.logging_config import get_logger
from utils.helpers import get_utc_now
from utils.exceptions import ModelServingError, ModelNotFoundError

logger = get_logger(__name__)
settings = get_settings()


class ModelServer:
    """
    High-performance model serving class.
    Handles model loading, caching, and prediction serving.
    """
    
    def __init__(self, model_name: str = "credit_risk_model"):
        """
        Initialize model server.
        
        Args:
            model_name: Name of model to serve
        """
        self.model_name = model_name
        self.model = None
        self.model_version = None
        self.feature_engineer = CreditRiskFeatureEngineer()
        self.prediction_cache = TTLCache(maxsize=1000, ttl=300)
        self.start_time = get_utc_now()
        self._load_model()
        
    def _load_model(self):
        """Load the latest active model."""
        try:
            pipeline = ModelTrainingPipeline()
            self.model = pipeline.load_latest_model(self.model_name)
            
            # Get model version from registry
            with get_db_session() as session:
                from utils.database import ModelRegistry
                latest = session.query(ModelRegistry).filter(
                    ModelRegistry.model_name == self.model_name,
                    ModelRegistry.status == "active"
                ).order_by(ModelRegistry.created_at.desc()).first()
                
                if latest:
                    self.model_version = latest.model_version
                    # Convert relative path to absolute path
                    if Path(latest.model_path).is_absolute():
                        model_path = Path(latest.model_path)
                    else:
                        # Get project root (2 levels up from src/models/)
                        project_root = Path(__file__).parent.parent.parent
                        model_path = project_root / latest.model_path
                    
                    # Load feature names from saved model artifacts
                    feature_names_path = model_path / "feature_names.json"
                    logger.info(f"Looking for feature names at: {feature_names_path}")
                    if feature_names_path.exists():
                        with open(feature_names_path, 'r') as f:
                            all_feature_names = json.load(f)
                            # Remove target variables from feature names
                            self.feature_engineer.feature_names = [
                                f for f in all_feature_names 
                                if f not in ['default_probability', 'loss_given_default', 'target', 'label']
                            ]
                            logger.info(f"Loaded {len(self.feature_engineer.feature_names)} feature names (removed {len(all_feature_names) - len(self.feature_engineer.feature_names)} target variables)")
                    else:
                        logger.error(f"Feature names file not found at: {feature_names_path}")
                    
                    # Load scalers if available
                    scaler_path = model_path / "scaler.joblib"
                    if scaler_path.exists():
                        self.feature_engineer.scalers['standard'] = joblib.load(scaler_path)
                    
                    logger.info(f"Loaded model version: {self.model_version}")
                else:
                    logger.warning("No model version found in registry")
                    
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def predict(
        self,
        credit_data: Dict[str, Any],
        use_cache: bool = True,
        log_prediction: bool = True
    ) -> Dict[str, Any]:
        """
        Make credit risk prediction.
        
        Args:
            credit_data: Credit application data
            use_cache: Whether to use prediction cache
            log_prediction: Whether to log prediction to database
        
        Returns:
            Prediction results with risk scores
        """
        start_time = time.time()
        request_id = self._generate_request_id(credit_data)
        
        try:
            # Check cache first
            if use_cache and request_id in self.prediction_cache:
                cached_result, cache_time = self.prediction_cache[request_id]
                if time.time() - cache_time < 300:
                    logger.info(f"Returning cached prediction for request {request_id}")
                    return cached_result
            
            # Validate input data (exclude request_id if present)
            data_for_validation = {k: v for k, v in credit_data.items() if k != 'request_id'}
            validation_result = validate_real_time_data([data_for_validation], 'credit')
            if not validation_result['is_valid']:
                return {
                    'success': False,
                    'error': 'Invalid credit data',
                    'validation_errors': validation_result['errors'],
                    'request_id': request_id
                }
            
            # Get current economic features
            economic_features = get_current_economic_features()
            if not economic_features:
                logger.warning("No economic features available, using defaults")
                economic_features = {
                    'fed_rate': 5.0,
                    'unemployment_rate': 4.0,
                    'economic_stress_index': 0.3
                }
            
            # Transform features for prediction (exclude request_id)
            features_array = self.feature_engineer.transform_prediction_data(
                data_for_validation,
                economic_features
            )
            
            # Make prediction
            if not self.feature_engineer.feature_names:
                logger.error("Feature names not loaded - using default feature names")
                # This should have been loaded from the saved model
                raise ValueError("Feature names not configured in feature engineer")
            
            logger.info(f"Features array shape: {features_array.shape}, Feature names count: {len(self.feature_engineer.feature_names)}")
            
            # Debug: Log the actual feature values
            logger.info(f"Raw input - Income: {data_for_validation.get('income')}, Employment: {data_for_validation.get('employment_length')}")
            logger.info(f"First 5 feature values after transformation: {features_array[:5]}")
            
            features_df = pd.DataFrame([features_array], columns=self.feature_engineer.feature_names)
            
            # Debug: Check if features are changing
            logger.info(f"Features DataFrame shape: {features_df.shape}")
            logger.info(f"Income-related features: {features_df[['income', 'income_to_loan_ratio']].values[0] if 'income' in features_df.columns else 'income not in columns'}")
            
            predictions = self.model.predict(features_df)
            
            # Extract results
            pd_score = float(predictions['probability_of_default'][0])
            lgd_score = float(predictions['loss_given_default'][0])
            el_score = float(predictions['expected_loss'][0])
            risk_rating = int(predictions['risk_rating'][0])
            
            # Create response
            response_time_ms = (time.time() - start_time) * 1000
            
            result = {
                'success': True,
                'request_id': request_id,
                'model_version': self.model_version,
                'predictions': {
                    'probability_of_default': pd_score,
                    'loss_given_default': lgd_score,
                    'expected_loss': el_score,
                    'risk_rating': risk_rating,
                    'risk_category': self._get_risk_category(risk_rating),
                    'decision': self._make_decision(pd_score, risk_rating)
                },
                'confidence_scores': {
                    'pd_confidence': self._calculate_confidence(pd_score),
                    'lgd_confidence': 0.85  # Placeholder - implement actual confidence
                },
                'economic_context': {
                    'fed_rate': economic_features.get('fed_rate'),
                    'unemployment_rate': economic_features.get('unemployment_rate'),
                    'economic_stress': economic_features.get('economic_stress_index')
                },
                'response_time_ms': response_time_ms,
                'timestamp': get_utc_now().isoformat()
            }
            
            # Cache result
            if use_cache:
                self.prediction_cache[request_id] = (result, time.time())
            
            # Log prediction
            if log_prediction:
                self._log_prediction(
                    request_id,
                    credit_data,
                    result['predictions'],
                    response_time_ms
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed for request {request_id}: {str(e)}")
            return {
                'success': False,
                'error': 'Prediction failed',
                'error_details': str(e),
                'request_id': request_id,
                'timestamp': get_utc_now().isoformat()
            }
    
    def batch_predict(
        self,
        credit_data_list: List[Dict[str, Any]],
        max_batch_size: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Make batch predictions for multiple credit applications.
        
        Args:
            credit_data_list: List of credit applications
            max_batch_size: Maximum batch size for processing
        
        Returns:
            List of prediction results
        """
        try:
            results = []
            total_items = len(credit_data_list)
            
            logger.info(f"Processing batch prediction for {total_items} items")
            
            # Process in batches
            for i in range(0, total_items, max_batch_size):
                batch = credit_data_list[i:i + max_batch_size]
                batch_results = []
                
                for credit_data in batch:
                    result = self.predict(
                        credit_data,
                        use_cache=True,
                        log_prediction=False  # Log separately for batch
                    )
                    batch_results.append(result)
                
                results.extend(batch_results)
                
                # Log progress
                processed = min(i + max_batch_size, total_items)
                logger.info(f"Processed {processed}/{total_items} items")
            
            # Log batch prediction summary
            success_count = sum(1 for r in results if r.get('success', False))
            logger.info(f"Batch prediction completed: {success_count}/{total_items} successful")
            
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}")
            return [{'success': False, 'error': str(e)} for _ in credit_data_list]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded model."""
        try:
            info = {
                'model_name': self.model_name,
                'model_version': self.model_version,
                'is_loaded': self.model is not None,
                'feature_count': len(self.feature_engineer.feature_names),
                'cache_size': len(self.prediction_cache),
                'server_uptime': (get_utc_now() - self.start_time).total_seconds()
            }
            
            # Get model metrics from registry
            if self.model_version:
                with get_db_session() as session:
                    from utils.database import ModelRegistry
                    model_record = session.query(ModelRegistry).filter(
                        ModelRegistry.model_name == self.model_name,
                        ModelRegistry.model_version == self.model_version
                    ).first()
                    
                    if model_record:
                        # Access attributes while session is active
                        if model_record.performance_metrics:
                            info['performance_metrics'] = json.loads(model_record.performance_metrics)
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get model info: {str(e)}")
            return {'error': str(e)}
    
    def reload_model(self) -> bool:
        """Reload the model (useful for model updates)."""
        try:
            logger.info("Reloading model...")
            self._load_model()
            self.prediction_cache.clear()  # Clear cache on reload
            logger.info("Model reloaded successfully")
            return True
        except Exception as e:
            logger.error(f"Model reload failed: {str(e)}")
            return False
    
    def _generate_request_id(self, credit_data: Dict[str, Any]) -> str:
        """Generate unique request ID for caching."""
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        # Create hash of input data
        clean_data = convert_types(credit_data)
        data_str = json.dumps(clean_data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _get_risk_category(self, risk_rating: int) -> str:
        """Convert numeric risk rating to category."""
        categories = {
            1: "Very Low Risk",
            2: "Low Risk",
            3: "Low-Medium Risk",
            4: "Medium Risk",
            5: "Medium Risk",
            6: "Medium-High Risk",
            7: "High Risk",
            8: "High Risk",
            9: "Very High Risk",
            10: "Extreme Risk"
        }
        return categories.get(risk_rating, "Unknown")
    
    def _make_decision(self, pd_score: float, risk_rating: int) -> Dict[str, Any]:
        """Make lending decision based on scores."""
        if pd_score < 0.05 and risk_rating <= 3:
            decision = "APPROVE"
            reason = "Low risk profile"
        elif pd_score < 0.15 and risk_rating <= 5:
            decision = "APPROVE_WITH_CONDITIONS"
            reason = "Moderate risk - additional conditions apply"
        elif pd_score < 0.30 and risk_rating <= 7:
            decision = "MANUAL_REVIEW"
            reason = "Elevated risk - requires manual underwriting"
        else:
            decision = "DECLINE"
            reason = "Risk exceeds acceptable threshold"
        
        return {
            'decision': decision,
            'reason': reason,
            'requires_review': decision == "MANUAL_REVIEW"
        }
    
    def _calculate_confidence(self, probability: float) -> float:
        """Calculate confidence score for probability prediction."""
        # Simple confidence based on distance from 0.5
        # More confident when probability is closer to 0 or 1
        distance_from_middle = abs(probability - 0.5)
        confidence = 0.5 + distance_from_middle
        return float(min(confidence, 0.95))  # Cap at 95%
    
    def _log_prediction(
        self,
        request_id: str,
        input_features: Dict[str, Any],
        predictions: Dict[str, Any],
        response_time_ms: float
    ):
        """Log prediction to database."""
        try:
            db_manager.log_prediction(
                request_id=request_id,
                model_name=self.model_name,
                model_version=self.model_version or "unknown",
                input_features=input_features,
                prediction_result=predictions,
                confidence_score=predictions.get('pd_confidence', 0.0),
                response_time_ms=response_time_ms
            )
        except Exception as e:
            logger.error(f"Failed to log prediction: {str(e)}")


# Global model server instance
_model_server = None


def get_model_server(model_name: str = "credit_risk_model") -> ModelServer:
    """
    Get or create model server instance (singleton pattern).
    
    Args:
        model_name: Name of model to serve
    
    Returns:
        ModelServer instance
    """
    global _model_server
    if _model_server is None or _model_server.model_name != model_name:
        _model_server = ModelServer(model_name)
    return _model_server


def predict_credit_risk(credit_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function for making predictions.
    
    Args:
        credit_data: Credit application data
    
    Returns:
        Prediction results
    """
    server = get_model_server()
    return server.predict(credit_data)


def warm_up_model_server():
    """Warm up model server by loading model and making test prediction."""
    try:
        logger.info("Warming up model server...")
        server = get_model_server()
        
        # Make dummy prediction to warm up
        test_data = {
            'customer_id': 'warmup_test',
            'loan_amount': 50000,
            'loan_term': 60,
            'interest_rate': 0.08,
            'income': 75000,
            'debt_to_income': 0.35,
            'credit_score': 720,
            'employment_length': 5,
            'home_ownership': 'MORTGAGE',
            'loan_purpose': 'DEBT_CONSOLIDATION'
        }
        
        result = server.predict(test_data, use_cache=False, log_prediction=False)
        
        if result.get('success'):
            logger.info("Model server warmed up successfully")
            return True
        else:
            logger.error(f"Model warmup failed: {result.get('error')}")
            return False
            
    except Exception as e:
        logger.error(f"Model warmup error: {str(e)}")
        return False