"""
Pydantic schemas for model management endpoints.
Defines data validation and serialization for model registry API.
"""

from pydantic import BaseModel, Field, validator, ConfigDict
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum


class ModelStatus(str, Enum):
    """Model status types."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"
    TRAINING = "training"
    FAILED = "failed"


class DriftStatus(str, Enum):
    """Model drift status levels."""
    STABLE = "stable"
    MINOR_DRIFT = "minor_drift"
    MODERATE_DRIFT = "moderate_drift"
    SIGNIFICANT_DRIFT = "significant_drift"
    INSUFFICIENT_DATA = "insufficient_data"


class ModelInfo(BaseModel):
    """
    Current model information schema.
    Provides details about the actively loaded model.
    """
    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "model_name": "credit_risk_model",
                "model_version": "v1.2.0",
                "is_loaded": True,
                "feature_count": 25,
                "performance_metrics": {
                    "auc_roc": 0.85,
                    "precision": 0.78,
                    "recall": 0.72,
                    "f1_score": 0.75
                },
                "training_config": {
                    "algorithm": "RandomForest",
                    "n_estimators": 100,
                    "max_depth": 10
                },
                "cache_size": 1500,
                "last_updated": "2024-01-15T09:30:00Z",
                "status": "active"
            }
        }
    )

    model_name: str = Field(..., description="Name of the model")
    model_version: str = Field(..., description="Version identifier")
    is_loaded: bool = Field(..., description="Whether model is currently loaded")
    feature_count: int = Field(..., description="Number of input features")
    performance_metrics: Dict[str, Any] = Field(..., description="Model performance metrics")
    training_config: Dict[str, Any] = Field(..., description="Training configuration used")
    cache_size: int = Field(..., description="Current prediction cache size")
    last_updated: Optional[str] = Field(None, description="Last update timestamp")
    status: ModelStatus = Field(..., description="Current model status")


class ModelPerformance(BaseModel):
    """
    Detailed model performance metrics schema.
    Comprehensive performance analysis for a specific model version.
    """
    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "model_version": "v1.2.0",
                "accuracy_metrics": {
                    "auc_roc": 0.85,
                    "auc_pr": 0.72,
                    "accuracy": 0.81,
                    "precision": 0.78,
                    "recall": 0.72,
                    "f1_score": 0.75,
                    "specificity": 0.83
                },
                "calibration_metrics": {
                    "brier_score": 0.12,
                    "calibration_slope": 0.95,
                    "calibration_intercept": 0.02,
                    "hosmer_lemeshow_p": 0.45
                },
                "business_metrics": {
                    "expected_loss_accuracy": 0.88,
                    "profit_optimization": 0.15,
                    "false_positive_rate": 0.08,
                    "false_negative_rate": 0.12
                },
                "validation_results": {
                    "cross_validation_score": 0.83,
                    "holdout_test_score": 0.82,
                    "bootstrap_confidence_interval": [0.80, 0.86]
                },
                "prediction_stats": {
                    "total_predictions": 15420,
                    "avg_response_time_ms": 45.3,
                    "cache_hit_rate": 0.35,
                    "error_rate": 0.002
                },
                "last_evaluated": "2024-01-15T08:00:00Z",
                "evaluation_period": "last_30_days"
            }
        }
    )

    model_version: str = Field(..., description="Model version identifier")
    accuracy_metrics: Dict[str, float] = Field(..., description="Accuracy-related metrics")
    calibration_metrics: Dict[str, float] = Field(..., description="Model calibration metrics")
    business_metrics: Dict[str, float] = Field(..., description="Business impact metrics")
    validation_results: Dict[str, Any] = Field(..., description="Validation test results")
    prediction_stats: Dict[str, Any] = Field(..., description="Recent prediction statistics")
    last_evaluated: str = Field(..., description="Last evaluation timestamp")
    evaluation_period: str = Field(..., description="Period of evaluation data")


class ModelRegistry(BaseModel):
    """
    Model registry entry schema.
    Represents a single model version in the registry.
    """
    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "model_name": "credit_risk_model",
                "model_version": "v1.2.0",
                "model_path": "/data/models/credit_risk_model_v1.2.0",
                "performance_metrics": {
                    "training_auc": 0.87,
                    "validation_auc": 0.85,
                    "test_auc": 0.83
                },
                "training_config": {
                    "algorithm": "RandomForest",
                    "features": 25,
                    "training_samples": 50000
                },
                "created_at": "2024-01-15T06:30:00Z",
                "updated_at": "2024-01-15T09:30:00Z",
                "status": "active",
                "description": "Production model with improved feature engineering"
            }
        }
    )

    model_name: str = Field(..., description="Name of the model")
    model_version: str = Field(..., description="Version identifier")
    model_path: str = Field(..., description="Path to model artifacts")
    performance_metrics: Dict[str, Any] = Field(..., description="Training performance metrics")
    training_config: Dict[str, Any] = Field(..., description="Training configuration")
    created_at: str = Field(..., description="Model creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    status: ModelStatus = Field(..., description="Current status")
    description: Optional[str] = Field(None, description="Model description")


class ModelComparisonRequest(BaseModel):
    """
    Model comparison request schema.
    Specifies which models to compare and what metrics to analyze.
    """
    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "model_versions": ["v1.1.0", "v1.2.0", "v1.3.0"],
                "metrics": ["auc_roc", "precision", "recall", "response_time_ms"],
                "comparison_type": "performance"
            }
        }
    )

    model_versions: List[str] = Field(
        ..., 
        description="List of model versions to compare",
        min_items=2,
        max_items=5
    )
    metrics: List[str] = Field(
        ...,
        description="List of metrics to compare",
        min_items=1
    )
    comparison_type: Optional[str] = Field(
        "performance",
        description="Type of comparison to perform"
    )
    
    @validator('model_versions')
    def validate_versions(cls, v):
        """Validate model versions format."""
        if len(set(v)) != len(v):
            raise ValueError('Model versions must be unique')
        return v
    
    @validator('metrics')
    def validate_metrics(cls, v):
        """Validate metric names."""
        valid_metrics = {
            'auc_roc', 'auc_pr', 'accuracy', 'precision', 'recall', 'f1_score',
            'brier_score', 'expected_loss_accuracy', 'response_time_ms'
        }
        invalid_metrics = set(v) - valid_metrics
        if invalid_metrics:
            raise ValueError(f'Invalid metrics: {invalid_metrics}')
        return v


class ModelComparisonResponse(BaseModel):
    """
    Model comparison response schema.
    Returns detailed comparison analysis between model versions.
    """
    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "models_compared": ["v1.1.0", "v1.2.0"],
                "comparison_metrics": ["auc_roc", "precision", "recall"],
                "results": [
                    {
                        "model_version": "v1.1.0",
                        "performance_metrics": {"auc_roc": 0.83, "precision": 0.75},
                        "status": "inactive"
                    },
                    {
                        "model_version": "v1.2.0",
                        "performance_metrics": {"auc_roc": 0.85, "precision": 0.78},
                        "status": "active"
                    }
                ],
                "analysis": {
                    "summary": "v1.2.0 shows improved performance across all metrics",
                    "best_performing": {
                        "auc_roc": {"model_version": "v1.2.0", "value": 0.85}
                    }
                },
                "recommendation": {
                    "deploy_version": "v1.2.0",
                    "confidence": 0.95,
                    "reasoning": "Significant improvement in key metrics with no trade-offs"
                },
                "comparison_timestamp": "2024-01-16T11:00:00Z"
            }
        }
    )

    models_compared: List[str] = Field(..., description="Model versions compared")
    comparison_metrics: List[str] = Field(..., description="Metrics used in comparison")
    results: List[Dict[str, Any]] = Field(..., description="Comparison results for each model")
    analysis: Dict[str, Any] = Field(..., description="Comparative analysis summary")
    recommendation: Dict[str, Any] = Field(..., description="Deployment recommendation")
    comparison_timestamp: str = Field(..., description="Comparison execution timestamp")


class DriftMetrics(BaseModel):
    """
    Model drift detection metrics schema.
    Measures statistical drift in model inputs and outputs.
    """
    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "model_version": "v1.2.0",
                "analysis_period_days": 30,
                "feature_drift": {
                    "credit_score": 0.02,
                    "income": 0.08,
                    "debt_to_income": 0.05
                },
                "prediction_drift": {
                    "pd_distribution_shift": 0.03
                },
                "performance_drift": {
                    "auc_roc_change": -0.015
                },
                "overall_drift_score": 0.045,
                "drift_status": "minor_drift",
                "recommendations": ["Monitor 'income' feature closely", "Consider partial retrain in 3 months"],
                "analysis_timestamp": "2024-02-15T10:00:00Z"
            }
        }
    )

    model_version: str = Field(..., description="Model version analyzed")
    analysis_period_days: int = Field(..., description="Number of days analyzed")
    feature_drift: Dict[str, float] = Field(..., description="Drift scores per feature")
    prediction_drift: Dict[str, float] = Field(..., description="Prediction distribution drift")
    performance_drift: Dict[str, float] = Field(..., description="Performance metric drift")
    overall_drift_score: float = Field(..., description="Overall drift severity score", ge=0, le=1)
    drift_status: DriftStatus = Field(..., description="Categorical drift assessment")
    recommendations: List[str] = Field(..., description="Recommended actions")
    analysis_timestamp: str = Field(..., description="Analysis execution timestamp")


class TrainingRequest(BaseModel):
    """
    Model training request schema.
    Specifies parameters for training a new model version.
    """
    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "model_name": "credit_risk_model_v2",
                "training_config": {
                    "algorithm": "XGBoost",
                    "learning_rate": 0.05,
                    "n_estimators": 200
                },
                "data_filters": {
                    "start_date": "2023-01-01",
                    "end_date": "2024-01-01"
                },
                "validation_split": 0.2,
                "description": "Retraining with new data and XGBoost algorithm",
                "auto_deploy": False
            }
        }
    )

    model_name: Optional[str] = Field("credit_risk_model", description="Name for the new model")
    training_config: Dict[str, Any] = Field(..., description="Training configuration parameters")
    data_filters: Optional[Dict[str, Any]] = Field(None, description="Data filtering criteria")
    validation_split: Optional[float] = Field(0.2, description="Validation set proportion", ge=0.1, le=0.4)
    description: Optional[str] = Field(None, description="Description of this training run")
    auto_deploy: Optional[bool] = Field(False, description="Automatically deploy if training succeeds")
    
    @validator('validation_split')
    def validate_split(cls, v):
        """Validate validation split is reasonable."""
        if v < 0.1 or v > 0.4:
            raise ValueError('Validation split must be between 0.1 and 0.4')
        return v


class TrainingResponse(BaseModel):
    """
    Model training response schema.
    Returns training job status and initial results.
    """
    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "training_job_id": "job_12345",
                "model_version": "v2.0.0_20240301100000",
                "status": "started",
                "estimated_completion_time": "2024-03-01T12:00:00Z",
                "initial_metrics": None,
                "message": "Training job started successfully. Monitor progress via /training/status/{job_id}",
                "started_at": "2024-03-01T10:00:00Z"
            }
        }
    )

    training_job_id: str = Field(..., description="Unique training job identifier")
    model_version: str = Field(..., description="Version assigned to new model")
    status: str = Field(..., description="Current training status")
    estimated_completion_time: Optional[str] = Field(None, description="Estimated completion time")
    initial_metrics: Optional[Dict[str, float]] = Field(None, description="Initial training metrics")
    message: str = Field(..., description="Status message")
    started_at: str = Field(..., description="Training start timestamp")