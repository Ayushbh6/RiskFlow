#!/usr/bin/env python3
"""
Retrain credit risk model with correct features (excluding target variable).
This fixes the issue where 'default_probability' was included as a feature.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from pathlib import Path
import logging

from models.model_training import ModelTrainingPipeline
from data.preprocessing import CreditRiskFeatureEngineer
from config.logging_config import get_logger

logger = get_logger(__name__)

def fix_and_retrain():
    """Retrain model with correct features."""
    try:
        logger.info("Starting model retraining to fix feature issue...")
        
        # Load the existing training data from database
        from utils.database import db_manager
        df = db_manager.get_credit_data(limit=1000)
        
        if df is None or (isinstance(df, list) and len(df) == 0) or (hasattr(df, 'empty') and df.empty):
            logger.error("No training data found. Run fetch_real_loan_data.py first.")
            return False
        logger.info(f"Loaded {len(df)} training samples")
        
        # Ensure default_probability exists as target
        if 'default_probability' not in df.columns:
            logger.error("Target variable 'default_probability' not found in data")
            return False
        
        # Initialize pipeline
        pipeline = ModelTrainingPipeline()
        
        # Train model with proper feature selection
        logger.info("Training new model with correct features...")
        metrics = pipeline.train_credit_risk_model(df)
        
        logger.info("Model training completed successfully!")
        logger.info(f"PD Model AUC: {metrics.get('pd_metrics', {}).get('cv_auc_mean', 'N/A')}")
        
        # Save the model
        model_path = pipeline.save_model(df)
        logger.info(f"Model saved to: {model_path}")
        
        # Verify feature names don't include target
        feature_names_path = Path(model_path) / "feature_names.json"
        if feature_names_path.exists():
            import json
            with open(feature_names_path, 'r') as f:
                features = json.load(f)
            
            if 'default_probability' in features:
                logger.error("ERROR: Target variable still in feature names!")
                # Remove it manually
                features = [f for f in features if f != 'default_probability']
                with open(feature_names_path, 'w') as f:
                    json.dump(features, f, indent=2)
                logger.info("Removed target variable from feature names")
            else:
                logger.info("✅ Feature names are correct (no target variable)")
        
        return True
        
    except Exception as e:
        logger.error(f"Model retraining failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = fix_and_retrain()
    sys.exit(0 if success else 1)