"""
Credit Risk Model implementation for RiskFlow MLOps Pipeline.
Implements Probability of Default (PD) and Loss Given Default (LGD) models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.calibration import CalibratedClassifierCV
import joblib
from datetime import datetime
import warnings
import os

from config.settings import get_settings
from config.logging_config import get_logger
from utils.exceptions import ModelError
from utils.helpers import get_utc_now

logger = get_logger(__name__)
settings = get_settings()


class ProbabilityOfDefaultModel:
    """
    Probability of Default (PD) model for credit risk assessment.
    Uses ensemble methods with calibration for accurate probability estimates.
    """
    
    def __init__(self, model_type: str = "ensemble"):
        """
        Initialize PD model.
        
        Args:
            model_type: Type of model ('logistic', 'random_forest', 'ensemble')
        """
        self.model_type = model_type
        self.model = None
        self.feature_importances_ = None
        self.is_fitted = False
        self.training_metrics = {}
        self.feature_names = []
        
    def _create_base_model(self):
        """Create base model based on model type."""
        if self.model_type == "logistic":
            return LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced',
                solver='lbfgs'
            )
        elif self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
        elif self.model_type == "ensemble":
            # Ensemble uses calibrated random forest for best performance
            base_rf = RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            return CalibratedClassifierCV(
                base_rf,
                method='isotonic',
                cv=3
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Train the PD model.
        
        Args:
            X_train: Training features
            y_train: Training labels (1 = default, 0 = no default)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        
        Returns:
            Dictionary of training metrics
        """
        try:
            logger.info(f"Training PD model ({self.model_type}) with {len(X_train)} samples")
            
            # Store feature names
            self.feature_names = list(X_train.columns)
            
            # Create and train model
            self.model = self._create_base_model()
            self.model.fit(X_train, y_train)
            
            # Calculate training metrics
            train_pred_proba = self.model.predict_proba(X_train)[:, 1]
            self.training_metrics['train_auc'] = roc_auc_score(y_train, train_pred_proba)
            
            # Cross-validation score
            cv_scores = cross_val_score(
                self.model, X_train, y_train,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='roc_auc',
                n_jobs=-1
            )
            self.training_metrics['cv_auc_mean'] = cv_scores.mean()
            self.training_metrics['cv_auc_std'] = cv_scores.std()
            
            # Validation metrics if provided
            if X_val is not None and y_val is not None:
                val_pred_proba = self.model.predict_proba(X_val)[:, 1]
                self.training_metrics['val_auc'] = roc_auc_score(y_val, val_pred_proba)
            
            # Extract feature importances
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importances_ = self.model.feature_importances_
            elif hasattr(self.model, 'base_estimator_') and hasattr(self.model.base_estimator_, 'feature_importances_'):
                # For calibrated models
                self.feature_importances_ = self.model.base_estimator_.feature_importances_
            
            self.is_fitted = True
            
            logger.info(f"PD model training completed. AUC: {self.training_metrics['cv_auc_mean']:.4f}")
            return self.training_metrics
            
        except Exception as e:
            logger.error(f"PD model training failed: {str(e)}")
            raise
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of default.
        
        Args:
            X: Features for prediction
        
        Returns:
            Array of default probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        try:
            # Handle missing columns (like default_probability that shouldn't be a feature)
            X_copy = X.copy()
            for feature in self.feature_names:
                if feature not in X_copy.columns:
                    if feature in ['default_probability', 'loss_given_default', 'target']:
                        # These are target variables, not features - add as 0
                        logger.warning(f"Target variable '{feature}' found in feature names, using 0")
                        X_copy[feature] = 0
                    else:
                        raise ValueError(f"Missing required feature: {feature}")
            
            # Ensure same feature order
            X_aligned = X_copy[self.feature_names]
            return self.model.predict_proba(X_aligned)[:, 1]
        except Exception as e:
            logger.error(f"PD prediction failed: {str(e)}")
            raise
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if self.feature_importances_ is None:
            return {}
        
        importance_dict = {}
        for feature, importance in zip(self.feature_names, self.feature_importances_):
            importance_dict[feature] = float(importance)
        
        # Sort by importance
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    def save(self, filepath: str) -> None:
        """Save model to disk."""
        try:
            model_data = {
                'model': self.model,
                'model_type': self.model_type,
                'feature_names': self.feature_names,
                'feature_importances': self.feature_importances_,
                'training_metrics': self.training_metrics,
                'is_fitted': self.is_fitted,
                'saved_at': get_utc_now().isoformat()
            }
            joblib.dump(model_data, filepath)
            logger.info(f"PD model saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save PD model: {str(e)}")
            raise
    
    def load(self, filepath: str) -> None:
        """Load model from disk."""
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.model_type = model_data['model_type']
            self.feature_names = model_data['feature_names']
            self.feature_importances_ = model_data['feature_importances']
            self.training_metrics = model_data['training_metrics']
            self.is_fitted = model_data['is_fitted']
            logger.info(f"PD model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load PD model: {str(e)}")
            raise


class LossGivenDefaultModel:
    """
    Loss Given Default (LGD) model for credit risk assessment.
    Estimates the percentage of loss if default occurs.
    """
    
    def __init__(self, model_type: str = "gradient_boosting"):
        """
        Initialize LGD model.
        
        Args:
            model_type: Type of model ('gradient_boosting', 'random_forest')
        """
        self.model_type = model_type
        self.model = None
        self.feature_importances_ = None
        self.is_fitted = False
        self.training_metrics = {}
        self.feature_names = []
        
    def _create_base_model(self):
        """Create base model based on model type."""
        if self.model_type == "gradient_boosting":
            return GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                subsample=0.8
            )
        elif self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Train the LGD model.
        
        Args:
            X_train: Training features
            y_train: Training LGD values (0-1 range)
            X_val: Validation features (optional)
            y_val: Validation LGD values (optional)
        
        Returns:
            Dictionary of training metrics
        """
        try:
            logger.info(f"Training LGD model ({self.model_type}) with {len(X_train)} samples")
            
            # Ensure LGD values are in [0, 1] range
            y_train = np.clip(y_train, 0, 1)
            
            # Store feature names
            self.feature_names = list(X_train.columns)
            
            # Create and train model
            self.model = self._create_base_model()
            self.model.fit(X_train, y_train)
            
            # Calculate training metrics
            train_pred = self.model.predict(X_train)
            train_pred = np.clip(train_pred, 0, 1)  # Ensure predictions in valid range
            
            self.training_metrics['train_rmse'] = np.sqrt(mean_squared_error(y_train, train_pred))
            self.training_metrics['train_mae'] = mean_absolute_error(y_train, train_pred)
            self.training_metrics['train_r2'] = r2_score(y_train, train_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(
                self.model, X_train, y_train,
                cv=KFold(n_splits=5, shuffle=True, random_state=42),
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            self.training_metrics['cv_rmse_mean'] = np.sqrt(-cv_scores.mean())
            self.training_metrics['cv_rmse_std'] = np.sqrt(cv_scores.std())
            
            # Validation metrics if provided
            if X_val is not None and y_val is not None:
                y_val = np.clip(y_val, 0, 1)
                val_pred = self.model.predict(X_val)
                val_pred = np.clip(val_pred, 0, 1)
                self.training_metrics['val_rmse'] = np.sqrt(mean_squared_error(y_val, val_pred))
                self.training_metrics['val_mae'] = mean_absolute_error(y_val, val_pred)
                self.training_metrics['val_r2'] = r2_score(y_val, val_pred)
            
            # Extract feature importances
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importances_ = self.model.feature_importances_
            
            self.is_fitted = True
            
            logger.info(f"LGD model training completed. RMSE: {self.training_metrics['cv_rmse_mean']:.4f}")
            return self.training_metrics
            
        except Exception as e:
            logger.error(f"LGD model training failed: {str(e)}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict loss given default.
        
        Args:
            X: Features for prediction
        
        Returns:
            Array of LGD values (0-1 range)
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        try:
            # Ensure same feature order
            X_aligned = X[self.feature_names]
            predictions = self.model.predict(X_aligned)
            # Clip predictions to valid range
            return np.clip(predictions, 0, 1)
        except Exception as e:
            logger.error(f"LGD prediction failed: {str(e)}")
            raise
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if self.feature_importances_ is None:
            return {}
        
        importance_dict = {}
        for feature, importance in zip(self.feature_names, self.feature_importances_):
            importance_dict[feature] = float(importance)
        
        # Sort by importance
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    def save(self, filepath: str) -> None:
        """Save model to disk."""
        try:
            model_data = {
                'model': self.model,
                'model_type': self.model_type,
                'feature_names': self.feature_names,
                'feature_importances': self.feature_importances_,
                'training_metrics': self.training_metrics,
                'is_fitted': self.is_fitted,
                'saved_at': get_utc_now().isoformat()
            }
            joblib.dump(model_data, filepath)
            logger.info(f"LGD model saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save LGD model: {str(e)}")
            raise
    
    def load(self, filepath: str) -> None:
        """Load model from disk."""
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.model_type = model_data['model_type']
            self.feature_names = model_data['feature_names']
            self.feature_importances_ = model_data['feature_importances']
            self.training_metrics = model_data['training_metrics']
            self.is_fitted = model_data['is_fitted']
            logger.info(f"LGD model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load LGD model: {str(e)}")
            raise


class CreditRiskModel:
    """
    Complete credit risk model combining PD and LGD models.
    Calculates Expected Loss (EL) = PD × LGD × EAD.
    """
    
    def __init__(
        self,
        pd_model_type: str = "ensemble",
        lgd_model_type: str = "gradient_boosting"
    ):
        """
        Initialize credit risk model.
        
        Args:
            pd_model_type: Type of PD model
            lgd_model_type: Type of LGD model
        """
        self.pd_model = ProbabilityOfDefaultModel(pd_model_type)
        self.lgd_model = LossGivenDefaultModel(lgd_model_type)
        self.is_fitted = False
        self.combined_metrics = {}
        
    def train(
        self,
        X_train: pd.DataFrame,
        y_train_default: pd.Series,
        y_train_lgd: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val_default: Optional[pd.Series] = None,
        y_val_lgd: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Train both PD and LGD models.
        
        Args:
            X_train: Training features
            y_train_default: Training default labels
            y_train_lgd: Training LGD values
            X_val: Validation features
            y_val_default: Validation default labels
            y_val_lgd: Validation LGD values
        
        Returns:
            Combined training metrics
        """
        try:
            logger.info("Training complete credit risk model (PD + LGD)")
            
            # Train PD model
            pd_metrics = self.pd_model.train(X_train, y_train_default, X_val, y_val_default)
            
            # Train LGD model (only on defaulted loans)
            default_mask_train = y_train_default == 1
            if default_mask_train.sum() > 0:
                lgd_metrics = self.lgd_model.train(
                    X_train[default_mask_train],
                    y_train_lgd[default_mask_train],
                    X_val[y_val_default == 1] if X_val is not None else None,
                    y_val_lgd[y_val_default == 1] if y_val_lgd is not None else None
                )
            else:
                logger.warning("No defaulted loans in training data for LGD model")
                lgd_metrics = {}
            
            # Combine metrics
            self.combined_metrics = {
                'pd_metrics': pd_metrics,
                'lgd_metrics': lgd_metrics,
                'training_samples': len(X_train),
                'default_rate': float(y_train_default.mean()),
                'avg_lgd': float(y_train_lgd[default_mask_train].mean()) if default_mask_train.sum() > 0 else 0.0
            }
            
            self.is_fitted = True
            
            logger.info("Credit risk model training completed successfully")
            return self.combined_metrics
            
        except Exception as e:
            logger.error(f"Credit risk model training failed: {str(e)}")
            raise
    
    def predict(
        self,
        X: pd.DataFrame,
        exposure_at_default: Optional[Union[float, np.ndarray]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Predict credit risk metrics.
        
        Args:
            X: Features for prediction
            exposure_at_default: EAD values (optional, defaults to 1.0)
        
        Returns:
            Dictionary with PD, LGD, and EL predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        try:
            # Predict PD
            pd_predictions = self.pd_model.predict_proba(X)
            
            # Predict LGD
            lgd_predictions = self.lgd_model.predict(X)
            
            # Calculate Expected Loss
            if exposure_at_default is None:
                exposure_at_default = 1.0
            
            expected_loss = pd_predictions * lgd_predictions * exposure_at_default
            
            return {
                'probability_of_default': pd_predictions,
                'loss_given_default': lgd_predictions,
                'expected_loss': expected_loss,
                'risk_rating': self._calculate_risk_rating(pd_predictions, lgd_predictions)
            }
            
        except Exception as e:
            logger.error(f"Credit risk prediction failed: {str(e)}")
            raise
    
    def _calculate_risk_rating(self, pd: np.ndarray, lgd: np.ndarray) -> np.ndarray:
        """
        Calculate risk rating based on PD and LGD.
        
        Returns rating from 1 (lowest risk) to 10 (highest risk).
        """
        # Combine PD and LGD for overall risk score
        risk_score = pd * 0.7 + lgd * 0.3  # PD weighted more heavily
        
        # Convert to 1-10 rating
        rating_bins = np.array([0, 0.02, 0.05, 0.1, 0.15, 0.25, 0.35, 0.5, 0.7, 0.85, 1.0])
        ratings = np.digitize(risk_score, rating_bins)
        
        return ratings
    
    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Get feature importance from both models."""
        return {
            'pd_importance': self.pd_model.get_feature_importance(),
            'lgd_importance': self.lgd_model.get_feature_importance()
        }
    
    def save(self, pd_filepath: str, lgd_filepath: str) -> None:
        """Save both models to disk."""
        self.pd_model.save(pd_filepath)
        self.lgd_model.save(lgd_filepath)
        logger.info("Credit risk models saved successfully")
    
    def load(self, pd_filepath: str, lgd_filepath: str) -> None:
        """Load both models from disk."""
        self.pd_model.load(pd_filepath)
        self.lgd_model.load(lgd_filepath)
        self.is_fitted = True
        logger.info("Credit risk models loaded successfully")

    def save_to_registry(self, model_name: str, model_version: str) -> str:
        """Save the model to a model registry."""
        # Placeholder for registry logic
        model_dir = f"models/{model_name}"
        os.makedirs(model_dir, exist_ok=True)
        
        pd_filepath = f"{model_dir}/{model_version}_pd.joblib"
        lgd_filepath = f"{model_dir}/{model_version}_lgd.joblib"
        self.save(pd_filepath, lgd_filepath)

        logger.info(f"Model saved to registry at {pd_filepath} and {lgd_filepath}")
        return pd_filepath


def create_default_credit_risk_model() -> CreditRiskModel:
    """Create a default credit risk model with optimal settings."""
    return CreditRiskModel(
        pd_model_type="ensemble",
        lgd_model_type="gradient_boosting"
    )

def save_model_to_registry(model: CreditRiskModel, model_name: str, model_version: str) -> str:
    """Helper function to save model."""
    return model.save_to_registry(model_name, model_version)

def load_model_from_registry(model_name: str, model_version: str) -> CreditRiskModel:
    """Helper function to load model."""
    model_dir = f"models/{model_name}"
    pd_filepath = f"{model_dir}/{model_version}_pd.joblib"
    lgd_filepath = f"{model_dir}/{model_version}_lgd.joblib"
    
    if not os.path.exists(pd_filepath) or not os.path.exists(lgd_filepath):
        raise ModelError(f"Model {model_name}:{model_version} not found in local registry.")
    
    model = CreditRiskModel()
    model.load(pd_filepath, lgd_filepath)
    return model