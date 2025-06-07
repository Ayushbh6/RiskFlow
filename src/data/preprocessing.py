"""
Feature engineering and preprocessing for RiskFlow Credit Risk MLOps Pipeline.
Works with real data only - transforms authentic data into ML-ready features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder, QuantileTransformer
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
import logging

from config.settings import get_settings
from config.logging_config import get_logger
from utils.exceptions import DataTransformationError
from config.settings import get_feature_config
from sklearn.model_selection import train_test_split
from utils.helpers import get_utc_now

logger = get_logger(__name__)
settings = get_settings()


class CreditRiskFeatureEngineer:
    """
    Feature engineering pipeline for credit risk modeling using real data.
    Transforms authentic credit and economic data into ML-ready features.
    """
    
    def __init__(self):
        self.feature_cache = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        
    def engineer_credit_features(self, credit_data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer credit risk features from real credit application data.
        
        Args:
            credit_data: DataFrame with real credit application data
        
        Returns:
            DataFrame with engineered features
        """
        try:
            logger.info(f"Engineering features for {len(credit_data)} credit records")
            
            if credit_data.empty:
                raise ValueError("Cannot engineer features from empty credit data")
            
            df = credit_data.copy()
            
            # Validate required columns
            required_cols = ['loan_amount', 'income', 'credit_score', 'debt_to_income']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # 1. Income-based features
            df['income_to_loan_ratio'] = df['income'] / (df['loan_amount'] + 1e-6)
            df['loan_to_income_ratio'] = df['loan_amount'] / (df['income'] + 1e-6)
            
            # Income quartiles based on real data distribution (handle single data point)
            try:
                df['income_quartile'] = pd.qcut(df['income'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
            except ValueError:
                # For single data points or insufficient unique values, assign based on income level
                df['income_quartile'] = df['income'].apply(lambda x: 
                    'Q1' if x < 40000 else 'Q2' if x < 60000 else 'Q3' if x < 90000 else 'Q4')
            
            # 2. Credit score features
            if 'credit_score' in df.columns:
                # FICO score bands (industry standard)
                df['credit_score_band'] = pd.cut(
                    df['credit_score'],
                    bins=[0, 580, 670, 740, 800, 850],
                    labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
                )
                
                # Credit utilization proxy
                df['credit_score_normalized'] = (df['credit_score'] - 300) / 550
            
            # 3. Debt-to-income features
            if 'debt_to_income' in df.columns:
                df['dti_risk_category'] = pd.cut(
                    df['debt_to_income'],
                    bins=[0, 0.2, 0.36, 0.5, 1.0, np.inf],
                    labels=['Low', 'Moderate', 'High', 'Very High', 'Extreme']
                )
                
                # High DTI flag (industry threshold)
                df['high_dti_flag'] = (df['debt_to_income'] > 0.43).astype(int)
            
            # 4. Loan characteristics
            if 'loan_term' in df.columns:
                # Loan term risk (longer terms = higher risk)
                df['loan_term_risk'] = df['loan_term'] / 360.0  # Normalized to 30-year max
                
                # Short-term loan flag
                df['short_term_loan'] = (df['loan_term'] <= 36).astype(int)
            
            if 'interest_rate' in df.columns:
                # Interest rate risk indicators
                df['high_interest_rate'] = (df['interest_rate'] > 0.15).astype(int)
                df['interest_rate_percentile'] = df['interest_rate'].rank(pct=True)
            
            # 5. Employment features
            if 'employment_length' in df.columns:
                # Employment stability
                df['employment_stability'] = np.where(
                    df['employment_length'] >= 2, 'Stable',
                    np.where(df['employment_length'] >= 1, 'Moderate', 'New')
                )
                
                # Long employment flag
                df['long_employment'] = (df['employment_length'] >= 5).astype(int)
            
            # 6. Home ownership features
            if 'home_ownership' in df.columns:
                # Ownership stability indicator
                df['ownership_stability'] = df['home_ownership'].map({
                    'OWN': 2, 'MORTGAGE': 1, 'RENT': 0, 'OTHER': 0
                }).fillna(0)
            
            # 7. Loan purpose risk
            if 'loan_purpose' in df.columns:
                # Risk scoring by loan purpose (based on industry data)
                purpose_risk_map = {
                    'DEBT_CONSOLIDATION': 0.12,  # Lower risk
                    'HOME_IMPROVEMENT': 0.08,
                    'MAJOR_PURCHASE': 0.10,
                    'CAR': 0.09,
                    'WEDDING': 0.15,
                    'VACATION': 0.18,
                    'MEDICAL': 0.11,
                    'CREDIT_CARD': 0.16,
                    'OTHER': 0.14
                }
                df['purpose_risk_score'] = df['loan_purpose'].map(purpose_risk_map).fillna(0.14)
            
            # 8. Composite risk scores
            # Calculate a composite financial stress indicator
            df['financial_stress_score'] = (
                df['debt_to_income'] * 0.4 +
                (1 - df['credit_score_normalized']) * 0.3 +
                df['loan_to_income_ratio'].clip(0, 2) * 0.3
            )
            
            # Payment capacity indicator
            if 'loan_term' in df.columns and 'interest_rate' in df.columns:
                # Monthly payment estimation
                monthly_rate = df['interest_rate'] / 12
                num_payments = df['loan_term']
                
                # PMT formula
                df['monthly_payment'] = df['loan_amount'] * (
                    monthly_rate * (1 + monthly_rate) ** num_payments
                ) / ((1 + monthly_rate) ** num_payments - 1)
                
                # Payment to income ratio
                df['payment_to_income_ratio'] = (df['monthly_payment'] * 12) / (df['income'] + 1e-6)
            
            logger.info(f"Engineered {df.shape[1] - credit_data.shape[1]} new features")
            return df
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {str(e)}")
            raise
    
    def engineer_economic_features(self, economic_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Engineer features from real economic indicators.
        
        Args:
            economic_data: Dictionary with real economic data
        
        Returns:
            Dictionary with engineered economic features
        """
        try:
            features = {}
            
            # Federal funds rate features
            if 'fed_funds_rate' in economic_data:
                fed_rate = float(economic_data['fed_funds_rate'])
                features['fed_rate'] = fed_rate
                features['fed_rate_high'] = int(fed_rate > 3.0)
                features['fed_rate_normalized'] = min(fed_rate / 10.0, 1.0)  # Cap at 10%
            
            # Unemployment rate features
            if 'unemployment_rate' in economic_data:
                unemployment = float(economic_data['unemployment_rate'])
                features['unemployment_rate'] = unemployment
                features['unemployment_high'] = int(unemployment > 6.0)
                features['unemployment_normalized'] = min(unemployment / 15.0, 1.0)  # Cap at 15%
            
            # Interest rate environment
            if 'avg_interest_rate' in economic_data:
                avg_rate = float(economic_data['avg_interest_rate'])
                features['avg_interest_rate'] = avg_rate
                features['high_rate_environment'] = int(avg_rate > 4.0)
            
            # Economic stress composite
            if len(features) >= 2:
                stress_components = []
                if 'unemployment_normalized' in features:
                    stress_components.append(features['unemployment_normalized'])
                if 'fed_rate_normalized' in features:
                    stress_components.append(features['fed_rate_normalized'])
                
                features['economic_stress_index'] = np.mean(stress_components)
                features['high_economic_stress'] = int(features['economic_stress_index'] > 0.5)
            
            logger.info(f"Engineered {len(features)} economic features")
            return features
            
        except Exception as e:
            logger.error(f"Economic feature engineering failed: {str(e)}")
            return {}
    
    def prepare_training_data(
        self,
        credit_df: pd.DataFrame,
        economic_features: Dict[str, float],
        target_column: str = 'is_default'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare final training dataset with all engineered features.
        
        Args:
            credit_df: DataFrame with credit features
            economic_features: Dictionary with economic features
            target_column: Name of target variable column
        
        Returns:
            Tuple of (features_df, target_series)
        """
        try:
            logger.info("Preparing training data")
            
            # Engineer credit features
            features_df = self.engineer_credit_features(credit_df)
            
            # Add economic features to each row
            for feature_name, feature_value in economic_features.items():
                features_df[f'eco_{feature_name}'] = feature_value
            
            # Separate features and target
            if target_column in features_df.columns:
                target = features_df[target_column].fillna(0).astype(int)  # Ensure no NaN values
                features = features_df.drop(columns=[target_column])
            else:
                logger.warning(f"Target column '{target_column}' not found")
                target = pd.Series(dtype=int)
                features = features_df
            
            # Remove non-feature columns and target variables
            columns_to_drop = ['id', 'customer_id', 'created_at', 'updated_at', 'loss_given_default', 'default_probability', 'target', 'label']
            columns_to_drop = [col for col in columns_to_drop if col in features.columns]
            if columns_to_drop:
                features = features.drop(columns=columns_to_drop)
                logger.info(f"Dropped columns: {columns_to_drop}")
            
            # Select only numeric features for ML
            numeric_features = features.select_dtypes(include=[np.number])
            
            # Handle missing values
            numeric_features = self._handle_missing_values(numeric_features)
            
            # Scale features
            scaled_features = self._scale_features(numeric_features)
            
            # Store feature names for later use
            self.feature_names = list(scaled_features.columns)
            
            logger.info(f"Prepared training data: {scaled_features.shape[0]} samples, {scaled_features.shape[1]} features")
            return scaled_features, target
            
        except Exception as e:
            logger.error(f"Training data preparation failed: {str(e)}")
            raise
    
    def transform_prediction_data(
        self,
        credit_data: Dict[str, Any],
        economic_features: Dict[str, float]
    ) -> np.ndarray:
        """
        Transform single credit application for prediction.
        
        Args:
            credit_data: Single credit application data
            economic_features: Current economic features
        
        Returns:
            Numpy array ready for model prediction
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame([credit_data])
            
            # Engineer features
            features_df = self.engineer_credit_features(df)
            
            # Add economic features
            for feature_name, feature_value in economic_features.items():
                features_df[f'eco_{feature_name}'] = feature_value
            
            # Select numeric features
            numeric_features = features_df.select_dtypes(include=[np.number])
            
            # Handle missing values
            numeric_features = self._handle_missing_values(numeric_features)
            
            # Debug: Log features before scaling
            logger.info(f"Features before scaling - Income: {numeric_features['income'].values[0] if 'income' in numeric_features else 'N/A'}")
            logger.info(f"Numeric features shape: {numeric_features.shape}")
            logger.info(f"Numeric features columns: {list(numeric_features.columns)[:10]}...")
            
            # Scale features using saved scalers
            scaled_features = self._scale_features(numeric_features, fit=False)
            
            # Debug: Log features after scaling
            logger.info(f"Features after scaling shape: {scaled_features.shape}")
            
            # Ensure same feature order as training
            if self.feature_names:
                # Reorder columns to match training data
                missing_features = set(self.feature_names) - set(scaled_features.columns)
                extra_features = set(scaled_features.columns) - set(self.feature_names)
                
                if missing_features:
                    logger.warning(f"Missing features for prediction: {missing_features}")
                    # Add missing features with zeros
                    for feature in missing_features:
                        # Skip target variable if it's in feature names by mistake
                        if feature in ['default_probability', 'loss_given_default', 'target', 'label']:
                            logger.warning(f"Skipping target variable '{feature}' in feature list")
                            continue
                        scaled_features[feature] = 0.0
                
                if extra_features:
                    logger.info(f"Dropping extra features: {extra_features}")
                    scaled_features = scaled_features.drop(columns=list(extra_features))
                
                # Reorder to match training
                scaled_features = scaled_features[self.feature_names]
            
            return scaled_features.values[0]
            
        except Exception as e:
            logger.error(f"Prediction data transformation failed: {str(e)}")
            raise
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in numeric features."""
        try:
            if df.isnull().sum().sum() == 0:
                return df
            
            # Use simple imputation for now
            imputer = SimpleImputer(strategy='median')
            imputed_values = imputer.fit_transform(df)
            
            return pd.DataFrame(imputed_values, columns=df.columns, index=df.index)
            
        except Exception as e:
            logger.warning(f"Missing value handling failed: {str(e)}")
            return df.fillna(0)
    
    def _scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numeric features."""
        try:
            if fit:
                # Fit new scaler
                scaler = StandardScaler()
                scaled_values = scaler.fit_transform(df)
                self.scalers['standard'] = scaler
            else:
                # Use existing scaler
                if 'standard' not in self.scalers:
                    logger.warning("No fitted scaler found, using identity scaling")
                    return df
                scaler = self.scalers['standard']
                scaled_values = scaler.transform(df)
            
            return pd.DataFrame(scaled_values, columns=df.columns, index=df.index)
            
        except Exception as e:
            logger.warning(f"Feature scaling failed: {str(e)}")
            return df


class RealTimeFeatureStore:
    """
    Feature store for real-time credit risk features.
    Maintains latest economic features for real-time scoring.
    """
    
    def __init__(self):
        self.latest_economic_features: Dict[str, Any] = {}
        self.feature_timestamp: Optional[datetime] = None
        self.engineer = CreditRiskFeatureEngineer()
    
    def update_economic_features(self, economic_data: Dict[str, Any]) -> bool:
        """
        Update the latest economic features.
        
        Args:
            economic_data: Real economic data dictionary
        
        Returns:
            True if updated successfully
        """
        try:
            features = self.engineer.engineer_economic_features(economic_data)
            
            if features:
                self.latest_economic_features = features
                self.feature_timestamp = get_utc_now()
                logger.info(f"Updated {len(features)} economic features")
                return True
            else:
                logger.warning("No economic features generated")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update economic features: {str(e)}")
            return False
    
    def get_latest_features(self) -> Tuple[Dict[str, float], Optional[datetime]]:
        """
        Get the latest economic features.
        
        Returns:
            Tuple of (features_dict, timestamp)
        """
        return self.latest_economic_features.copy(), self.feature_timestamp
    
    def is_features_fresh(self, max_age_hours: int = 6) -> bool:
        """
        Check if features are fresh enough for real-time use.
        
        Args:
            max_age_hours: Maximum age in hours
        
        Returns:
            True if features are fresh
        """
        if not self.feature_timestamp:
            return False
        
        age = get_utc_now() - self.feature_timestamp
        return (age.total_seconds() / 3600) <= max_age_hours

    def get_feature_names(self) -> List[str]:
        """
        Get the feature names for the latest economic features.
        
        Returns:
            List of feature names
        """
        return list(self.latest_economic_features.keys())


# Global feature store instance
feature_store = RealTimeFeatureStore()


def get_current_economic_features() -> Dict[str, float]:
    """
    Get current economic features for real-time scoring.
    
    Returns:
        Dictionary with current economic features
    """
    features, timestamp = feature_store.get_latest_features()
    
    if not feature_store.is_features_fresh():
        logger.warning("Economic features are stale - may impact prediction accuracy")
    
    return features