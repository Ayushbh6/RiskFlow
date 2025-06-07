"""
Model training pipeline with MLflow tracking for RiskFlow Credit Risk MLOps.
Handles model training, experiment tracking, and model registry.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from datetime import datetime
import json
import os
from pathlib import Path

from models.credit_risk_model import CreditRiskModel, ProbabilityOfDefaultModel, LossGivenDefaultModel
from data.preprocessing import CreditRiskFeatureEngineer, get_current_economic_features
from utils.database import db_manager, get_db_session
from config.settings import get_settings
from config.logging_config import get_logger
from data.ingestion import DataIngestionPipeline
from utils.database import DatabaseManager
from utils.exceptions import ModelTrainingError
from utils.helpers import get_utc_now

logger = get_logger(__name__)
settings = get_settings()


class ModelTrainingPipeline:
    """
    Complete model training pipeline with MLflow tracking.
    Handles data preparation, model training, and experiment logging.
    """
    
    def __init__(self):
        """Initialize training pipeline."""
        self.settings = settings
        self.feature_engineer = CreditRiskFeatureEngineer()
        self.params = {
            'model_type': 'classification',
            'random_state': 42,
            'n_estimators': 100,
            'max_depth': 10
        }
        self.model_name = "credit_risk_model"
        self.db_manager = db_manager
        self.model = CreditRiskModel(pd_model_type='ensemble', lgd_model_type='gradient_boosting')
        self.mlflow_client = None
        self._setup_mlflow()
        
    def _setup_mlflow(self):
        """Set up MLflow tracking."""
        try:
            # Set tracking URI
            mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
            
            # Create experiment if it doesn't exist
            try:
                mlflow.create_experiment(self.settings.mlflow_experiment_name)
            except Exception:
                # Experiment already exists
                pass
            
            mlflow.set_experiment(self.settings.mlflow_experiment_name)
            self.mlflow_client = MlflowClient()
            
            logger.info(f"MLflow tracking initialized at {self.settings.mlflow_tracking_uri}")
            
        except Exception as e:
            logger.error(f"MLflow setup failed: {str(e)}")
            raise
    
    def prepare_training_data(
        self,
        limit: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare training data from database.
        
        Args:
            limit: Maximum number of records to use
        
        Returns:
            Tuple of (features, default_labels, lgd_values)
        """
        try:
            logger.info("Preparing training data from database")
            
            # Get credit data from database
            credit_data = db_manager.get_credit_data(limit=limit)
            
            if not credit_data:
                raise ValueError("No credit data available for training")
            
            credit_df = pd.DataFrame(credit_data)
            
            # Get current economic features
            economic_features = get_current_economic_features()
            
            if not economic_features:
                logger.warning("No economic features available, using defaults")
                economic_features = {
                    'fed_rate': 5.0,
                    'unemployment_rate': 4.0,
                    'economic_stress_index': 0.3
                }
            
            # Prepare features and targets
            features, default_labels = self.feature_engineer.prepare_training_data(
                credit_df,
                economic_features,
                target_column='is_default'
            )
            
            # Get LGD values
            lgd_values = credit_df['loss_given_default'].fillna(0.4)  # Default LGD of 40%
            
            logger.info(f"Prepared {len(features)} samples for training")
            logger.info(f"Default rate: {default_labels.mean():.2%}")
            logger.info(f"Average LGD: {lgd_values.mean():.2%}")
            
            return features, default_labels, lgd_values
            
        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}")
            raise
    
    def train_credit_risk_model(
        self,
        features: pd.DataFrame,
        default_labels: pd.Series,
        lgd_values: pd.Series,
        test_size: float = 0.2,
        pd_model_type: str = "ensemble",
        lgd_model_type: str = "gradient_boosting",
        track_experiment: bool = True
    ) -> Tuple[CreditRiskModel, Dict[str, Any]]:
        """
        Train credit risk model with MLflow tracking.
        
        Args:
            features: Feature DataFrame
            default_labels: Default labels (0/1)
            lgd_values: LGD values (0-1)
            test_size: Validation split ratio
            pd_model_type: Type of PD model
            lgd_model_type: Type of LGD model
            track_experiment: Whether to track with MLflow
        
        Returns:
            Tuple of (trained_model, metrics)
        """
        try:
            # Split data
            X_train, X_val, y_default_train, y_default_val, y_lgd_train, y_lgd_val = train_test_split(
                features, default_labels, lgd_values,
                test_size=test_size,
                random_state=42,
                stratify=default_labels
            )
            
            logger.info(f"Training set: {len(X_train)} samples")
            logger.info(f"Validation set: {len(X_val)} samples")
            
            # Start MLflow run
            if track_experiment:
                mlflow.start_run()
                
                # Log parameters
                mlflow.log_param("pd_model_type", pd_model_type)
                mlflow.log_param("lgd_model_type", lgd_model_type)
                mlflow.log_param("training_samples", len(X_train))
                mlflow.log_param("validation_samples", len(X_val))
                mlflow.log_param("feature_count", X_train.shape[1])
                mlflow.log_param("default_rate_train", float(y_default_train.mean()))
                
            # Create and train model
            model = CreditRiskModel(pd_model_type, lgd_model_type)
            metrics = model.train(
                X_train, y_default_train, y_lgd_train,
                X_val, y_default_val, y_lgd_val
            )
            
            # Calculate additional validation metrics
            val_predictions = model.predict(X_val)
            
            # Expected loss metrics
            val_el = val_predictions['expected_loss']
            metrics['val_expected_loss_mean'] = float(val_el.mean())
            metrics['val_expected_loss_std'] = float(val_el.std())
            
            # Risk rating distribution
            risk_ratings = val_predictions['risk_rating']
            rating_dist = pd.Series(risk_ratings).value_counts(normalize=True).to_dict()
            metrics['risk_rating_distribution'] = rating_dist
            
            if track_experiment:
                # Log metrics
                self._log_metrics_to_mlflow(metrics)
                
                # Log feature importance
                importance = model.get_feature_importance()
                mlflow.log_dict(importance, "feature_importance.json")
                
                # Log model
                mlflow.sklearn.log_model(
                    model.pd_model.model,
                    "pd_model",
                    registered_model_name="credit_risk_pd_model"
                )
                mlflow.sklearn.log_model(
                    model.lgd_model.model,
                    "lgd_model",
                    registered_model_name="credit_risk_lgd_model"
                )
                
                # Save run ID
                run_id = mlflow.active_run().info.run_id
                metrics['mlflow_run_id'] = run_id
                
                mlflow.end_run()
                
                logger.info(f"Model training completed. MLflow run ID: {run_id}")
            else:
                logger.info("Model training completed without MLflow tracking")
            
            return model, metrics
            
        except Exception as e:
            if track_experiment and mlflow.active_run():
                mlflow.end_run(status="FAILED")
            logger.error(f"Model training failed: {str(e)}")
            raise
    
    def _log_metrics_to_mlflow(self, metrics: Dict[str, Any]):
        """Log metrics to MLflow."""
        try:
            # Log PD metrics
            if 'pd_metrics' in metrics:
                for key, value in metrics['pd_metrics'].items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"pd_{key}", value)
            
            # Log LGD metrics
            if 'lgd_metrics' in metrics:
                for key, value in metrics['lgd_metrics'].items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"lgd_{key}", value)
            
            # Log combined metrics
            for key, value in metrics.items():
                if key not in ['pd_metrics', 'lgd_metrics', 'risk_rating_distribution'] and isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)
            
        except Exception as e:
            logger.error(f"Failed to log metrics to MLflow: {str(e)}")
    
    def register_model(
        self,
        model: CreditRiskModel,
        metrics: Dict[str, Any],
        model_name: str = "credit_risk_model",
        model_version: str = None
    ) -> int:
        """
        Register model in database and save artifacts.
        
        Args:
            model: Trained credit risk model
            metrics: Model metrics
            model_name: Name for model registration
            model_version: Version string (auto-generated if None)
        
        Returns:
            Model registry ID
        """
        try:
            # Generate version if not provided
            if model_version is None:
                model_version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            
            # Create model directory with absolute path
            base_path = Path(__file__).parent.parent.parent  # Get to project root
            model_dir = base_path / "data" / "models" / model_name / model_version
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model artifacts
            pd_path = str(model_dir / "pd_model.joblib")
            lgd_path = str(model_dir / "lgd_model.joblib")
            model.save(pd_path, lgd_path)
            
            # Save feature names
            feature_names_path = model_dir / "feature_names.json"
            with open(feature_names_path, 'w') as f:
                json.dump(self.feature_engineer.feature_names, f, indent=2)
            
            # Save scaler if available
            if 'standard' in self.feature_engineer.scalers:
                import joblib
                scaler_path = model_dir / "scaler.joblib"
                joblib.dump(self.feature_engineer.scalers['standard'], scaler_path)
            
            # Save metrics
            metrics_path = model_dir / "metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Register in database
            registry_id = db_manager.register_model(
                model_name=model_name,
                model_version=model_version,
                model_type="credit_risk",
                model_path=str(model_dir),
                mlflow_run_id=metrics.get('mlflow_run_id', ''),
                performance_metrics={
                    'pd_auc': metrics.get('pd_metrics', {}).get('cv_auc_mean', 0),
                    'lgd_rmse': metrics.get('lgd_metrics', {}).get('cv_rmse_mean', 0),
                    'expected_loss_mean': metrics.get('val_expected_loss_mean', 0)
                }
            )
            
            logger.info(f"Model registered with ID: {registry_id}")
            return registry_id
            
        except Exception as e:
            logger.error(f"Model registration failed: {str(e)}")
            raise
    
    def load_latest_model(self, model_name: str = "credit_risk_model") -> CreditRiskModel:
        """
        Load the latest registered model.
        
        Args:
            model_name: Name of the model to load
        
        Returns:
            Loaded credit risk model
        """
        try:
            # Query latest active model from registry
            with get_db_session() as session:
                from utils.database import ModelRegistry
                latest = session.query(ModelRegistry).filter(
                    ModelRegistry.model_name == model_name,
                    ModelRegistry.status == "active"
                ).order_by(ModelRegistry.created_at.desc()).first()
                
                if not latest:
                    raise ValueError(f"No active model found for {model_name}")
                
                model_path = Path(latest.model_path)
                model_version = latest.model_version
                
                # If path is relative, make it absolute
                if not model_path.is_absolute():
                    base_path = Path(__file__).parent.parent.parent
                    model_path = base_path / model_path
            
            # Load model
            model = CreditRiskModel()
            model.load(
                str(model_path / "pd_model.joblib"),
                str(model_path / "lgd_model.joblib")
            )
            
            logger.info(f"Loaded model version {model_version}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load latest model: {str(e)}")
            raise
    
    def run_training_experiment(
        self,
        experiment_name: Optional[str] = None,
        data_limit: Optional[int] = None,
        pd_model_types: List[str] = None,
        lgd_model_types: List[str] = None
    ) -> Dict[str, Any]:
        """
        Run complete training experiment with multiple model types.
        
        Args:
            experiment_name: MLflow experiment name
            data_limit: Limit on training data
            pd_model_types: List of PD model types to try
            lgd_model_types: List of LGD model types to try
        
        Returns:
            Experiment results
        """
        try:
            if experiment_name:
                mlflow.set_experiment(experiment_name)
            
            # Default model types
            if pd_model_types is None:
                pd_model_types = ["ensemble", "random_forest", "logistic"]
            if lgd_model_types is None:
                lgd_model_types = ["gradient_boosting", "random_forest"]
            
            # Prepare data once
            features, default_labels, lgd_values = self.prepare_training_data(limit=data_limit)
            
            results = []
            best_model = None
            best_score = 0
            
            # Try different model combinations
            for pd_type in pd_model_types:
                for lgd_type in lgd_model_types:
                    logger.info(f"Training combination: PD={pd_type}, LGD={lgd_type}")
                    
                    try:
                        model, metrics = self.train_credit_risk_model(
                            features, default_labels, lgd_values,
                            pd_model_type=pd_type,
                            lgd_model_type=lgd_type
                        )
                        
                        # Calculate combined score
                        pd_score = metrics.get('pd_metrics', {}).get('cv_auc_mean', 0)
                        lgd_score = 1 - metrics.get('lgd_metrics', {}).get('cv_rmse_mean', 1)
                        combined_score = pd_score * 0.7 + lgd_score * 0.3
                        
                        result = {
                            'pd_type': pd_type,
                            'lgd_type': lgd_type,
                            'combined_score': combined_score,
                            'metrics': metrics,
                            'model': model
                        }
                        results.append(result)
                        
                        if combined_score > best_score:
                            best_score = combined_score
                            best_model = result
                            
                    except Exception as e:
                        logger.error(f"Failed to train {pd_type}/{lgd_type}: {str(e)}")
                        continue
            
            # Register best model
            if best_model:
                logger.info(f"Best model: PD={best_model['pd_type']}, LGD={best_model['lgd_type']}")
                registry_id = self.register_model(
                    best_model['model'],
                    best_model['metrics']
                )
                best_model['registry_id'] = registry_id
            
            return {
                'results': results,
                'best_model': best_model,
                'experiment_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Training experiment failed: {str(e)}")
            raise

    def run_training_pipeline(self, test_size: float = 0.2, random_state: int = 42) -> str:
        """
        Run the full model training pipeline.
        
        Args:
            test_size: Proportion of data to use for testing
            random_state: Random state for reproducibility
            
        Returns:
            The version of the newly trained model
        """
        try:
            logger.info("Starting model training pipeline...")
            
            # Generate a unique model version
            model_version = f"v{get_utc_now().strftime('%Y%m%d%H%M%S')}"
            logger.info(f"Generated model version: {model_version}")

            # 1. Data Ingestion
            logger.info("Step 1: Ingesting data...")
            raw_data = self.ingest_data()
            
            # 2. Feature Engineering
            logger.info("Step 2: Engineering features...")
            features, target = self.feature_engineer.fit_transform(raw_data)
            
            # 3. Data Splitting
            logger.info("Step 3: Splitting data...")
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=test_size, random_state=random_state, stratify=target
            )
            
            # 4. Model Training
            logger.info("Step 4: Training model...")
            self.model.train(X_train, y_train)
            
            # 5. Model Evaluation
            logger.info("Step 5: Evaluating model...")
            metrics = self.model.evaluate(X_test, y_test)
            feature_importance = self.model.get_feature_importance()
            
            # 6. Log experiment
            logger.info("Step 6: Logging experiment...")
            self.log_experiment(
                model_version=model_version,
                metrics=metrics,
                params=self.params,
                feature_importance=feature_importance
            )
            
            logger.info("Model training pipeline completed successfully")
            return model_version
            
        except Exception as e:
            logger.error(f"Model training pipeline failed: {e}")
            raise ModelTrainingError(f"Training pipeline failed: {str(e)}")

    def ingest_data(self) -> pd.DataFrame:
        # For this example, we'll get data from the DB.
        # In a real scenario, this might trigger a DataIngestionPipeline job.
        return self._get_latest_data_from_db()

    def log_experiment(
        self,
        model_version: str,
        metrics: Dict[str, Any],
        params: Dict[str, Any],
        feature_importance: List[Dict[str, Any]]
    ) -> None:
        """Log training experiment details."""
        experiment_data = {
            "model_name": self.model_name,
            "model_version": model_version,
            "metrics": metrics,
            "parameters": params,
            "feature_importance": feature_importance,
            "experiment_timestamp": get_utc_now().isoformat()
        }
        self.db_manager.log_training_experiment(experiment_data)
        logger.info(f"Logged experiment for model version {model_version}")

    def _get_latest_data_from_db(self) -> pd.DataFrame:
        # Placeholder to get data from the database
        logger.info("Fetching latest data from database for training...")
        # This should be implemented to pull a relevant training dataset
        return self.db_manager.get_training_data(limit=10000)


def train_default_model() -> Tuple[CreditRiskModel, Dict[str, Any]]:
    """
    Train a default credit risk model with standard settings.
    
    Returns:
        Tuple of (trained_model, metrics)
    """
    pipeline = ModelTrainingPipeline()
    features, default_labels, lgd_values = pipeline.prepare_training_data()
    return pipeline.train_credit_risk_model(features, default_labels, lgd_values)