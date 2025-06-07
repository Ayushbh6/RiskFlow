"""
Model management endpoints for RiskFlow Credit Risk API.
Provides model registry, versioning, and performance monitoring.
"""

from fastapi import APIRouter, HTTPException, Depends, status, Query
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json

from api.schemas.model import (
    ModelInfo,
    ModelPerformance,
    ModelRegistry,
    ModelComparisonRequest,
    ModelComparisonResponse
)
from models.model_serving import get_model_server
from models.model_training import ModelTrainingPipeline
from utils.database import db_manager, get_db_session
from config.settings import get_settings
from config.logging_config import get_logger
from utils.exceptions import APIError, ModelNotFoundError
from utils.helpers import get_utc_now

logger = get_logger(__name__)
settings = get_settings()
router = APIRouter()


@router.get("/info", response_model=ModelInfo)
async def get_current_model_info():
    """
    Get information about the currently active model.
    
    Returns model version, performance metrics, and status.
    """
    try:
        model_server = get_model_server()
        model_info = model_server.get_model_info()
        
        if 'error' in model_info:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "Model info retrieval failed",
                    "message": model_info['error']
                }
            )
        
        # Get additional model details from registry
        with get_db_session() as session:
            from utils.database import ModelRegistry as ModelRegistryTable
            model_record = session.query(ModelRegistryTable).filter(
                ModelRegistryTable.model_name == model_info.get('model_name', 'credit_risk_model'),
                ModelRegistryTable.status == "active"
            ).order_by(ModelRegistryTable.created_at.desc()).first()
            
            performance_metrics = {}
            training_config = {}
            
            if model_record:
                if model_record.performance_metrics:
                    performance_metrics = json.loads(model_record.performance_metrics)
                if model_record.training_config:
                    training_config = json.loads(model_record.training_config)
        
        response = ModelInfo(
            model_name=model_info.get('model_name', 'unknown'),
            model_version=model_info.get('model_version', 'unknown'),
            is_loaded=model_info.get('is_loaded', False),
            feature_count=model_info.get('feature_count', 0),
            performance_metrics=performance_metrics,
            training_config=training_config,
            cache_size=model_info.get('cache_size', 0),
            last_updated=model_record.updated_at.isoformat() if model_record else None,
            status="active" if model_info.get('is_loaded') else "inactive"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving model info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Model info error",
                "message": str(e)
            }
        )


@router.get("/registry", response_model=List[ModelRegistry])
async def get_model_registry(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status_filter: Optional[str] = Query(None, regex="^(active|inactive|archived)$")
):
    """
    Get model registry with version history.
    
    Returns list of all model versions with metadata.
    """
    try:
        with get_db_session() as session:
            from ...utils.database import ModelRegistry as ModelRegistryTable
            
            query = session.query(ModelRegistryTable)
            
            # Apply status filter if provided
            if status_filter:
                query = query.filter(ModelRegistryTable.status == status_filter)
            
            # Apply pagination
            models = query.order_by(ModelRegistryTable.created_at.desc()).offset(offset).limit(limit).all()
            
            # Convert to response format
            model_list = []
            for model in models:
                performance_metrics = {}
                training_config = {}
                
                if model.performance_metrics:
                    performance_metrics = json.loads(model.performance_metrics)
                if model.training_config:
                    training_config = json.loads(model.training_config)
                
                model_list.append(ModelRegistry(
                    model_name=model.model_name,
                    model_version=model.model_version,
                    model_path=model.model_path,
                    performance_metrics=performance_metrics,
                    training_config=training_config,
                    created_at=model.created_at.isoformat(),
                    updated_at=model.updated_at.isoformat(),
                    status=model.status,
                    description=model.description
                ))
        
        return model_list
        
    except Exception as e:
        logger.error(f"Error retrieving model registry: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Registry retrieval error",
                "message": str(e)
            }
        )


@router.get("/performance/{model_version}", response_model=ModelPerformance)
async def get_model_performance(model_version: str):
    """
    Get detailed performance metrics for a specific model version.
    
    Returns comprehensive performance analysis and validation results.
    """
    try:
        with get_db_session() as session:
            from ...utils.database import ModelRegistry as ModelRegistryTable
            
            model_record = session.query(ModelRegistryTable).filter(
                ModelRegistryTable.model_version == model_version
            ).first()
            
            if not model_record:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={
                        "error": "Model not found",
                        "message": f"No model found with version: {model_version}"
                    }
                )
            
            # Parse performance metrics
            performance_metrics = {}
            if model_record.performance_metrics:
                performance_metrics = json.loads(model_record.performance_metrics)
            
            # Get recent prediction statistics
            prediction_stats = db_manager.get_model_prediction_stats(
                model_version,
                days=30
            )
            
            response = ModelPerformance(
                model_version=model_version,
                accuracy_metrics=performance_metrics.get('accuracy_metrics', {}),
                calibration_metrics=performance_metrics.get('calibration_metrics', {}),
                business_metrics=performance_metrics.get('business_metrics', {}),
                validation_results=performance_metrics.get('validation_results', {}),
                prediction_stats=prediction_stats,
                last_evaluated=model_record.updated_at.isoformat(),
                evaluation_period="training_validation"
            )
            
            return response
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving model performance for {model_version}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Performance retrieval error",
                "message": str(e)
            }
        )


@router.post("/compare", response_model=ModelComparisonResponse)
async def compare_models(request: ModelComparisonRequest):
    """
    Compare performance between multiple model versions.
    
    Returns side-by-side comparison of model metrics and recommendations.
    """
    try:
        if len(request.model_versions) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least 2 model versions required for comparison"
            )
        
        if len(request.model_versions) > 5:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot compare more than 5 model versions at once"
            )
        
        comparison_results = []
        
        with get_db_session() as session:
            from ...utils.database import ModelRegistry as ModelRegistryTable
            
            for version in request.model_versions:
                model_record = session.query(ModelRegistryTable).filter(
                    ModelRegistryTable.model_version == version
                ).first()
                
                if not model_record:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Model version {version} not found"
                    )
                
                # Parse performance metrics
                performance_metrics = {}
                if model_record.performance_metrics:
                    performance_metrics = json.loads(model_record.performance_metrics)
                
                # Get prediction statistics
                prediction_stats = db_manager.get_model_prediction_stats(version, days=30)
                
                comparison_results.append({
                    "model_version": version,
                    "performance_metrics": performance_metrics,
                    "prediction_stats": prediction_stats,
                    "created_at": model_record.created_at.isoformat(),
                    "status": model_record.status
                })
        
        # Generate comparison analysis
        analysis = _generate_model_comparison_analysis(comparison_results, request.metrics)
        
        response = ModelComparisonResponse(
            models_compared=request.model_versions,
            comparison_metrics=request.metrics,
            results=comparison_results,
            analysis=analysis,
            recommendation=_generate_model_recommendation(comparison_results),
            comparison_timestamp=get_utc_now().isoformat()
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Model comparison error",
                "message": str(e)
            }
        )


@router.post("/reload")
async def reload_model():
    """
    Reload the currently active model.
    
    Useful for deploying new model versions without restarting the service.
    """
    try:
        model_server = get_model_server()
        success = model_server.reload_model()
        
        if success:
            # Get updated model info
            model_info = model_server.get_model_info()
            
            return {
                "success": True,
                "message": "Model reloaded successfully",
                "model_version": model_info.get('model_version'),
                "timestamp": get_utc_now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "Model reload failed",
                    "message": "Failed to reload model - check logs for details"
                }
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reloading model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Model reload error",
                "message": str(e)
            }
        )


@router.put("/activate/{model_version}")
async def activate_model_version(model_version: str):
    """
    Activate a specific model version.
    
    Sets the specified model version as active and reloads the model server.
    """
    try:
        with get_db_session() as session:
            from ...utils.database import ModelRegistry as ModelRegistryTable
            
            # Check if model version exists
            model_record = session.query(ModelRegistryTable).filter(
                ModelRegistryTable.model_version == model_version
            ).first()
            
            if not model_record:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model version {model_version} not found"
                )
            
            # Deactivate all other models
            session.query(ModelRegistryTable).filter(
                ModelRegistryTable.model_name == model_record.model_name
            ).update({"status": "inactive"})
            
            # Activate the specified model
            model_record.status = "active"
            model_record.updated_at = get_utc_now()
            session.commit()
        
        # Reload model server with new active model
        model_server = get_model_server()
        reload_success = model_server.reload_model()
        
        if not reload_success:
            logger.warning(f"Model version {model_version} activated but reload failed")
        
        return {
            "success": True,
            "message": f"Model version {model_version} activated successfully",
            "model_version": model_version,
            "reload_success": reload_success,
            "timestamp": get_utc_now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error activating model version {model_version}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Model activation error",
                "message": str(e)
            }
        )


@router.get("/metrics/drift")
async def get_model_drift_metrics(
    days: int = Query(30, ge=1, le=365),
    model_version: Optional[str] = None
):
    """
    Get model drift detection metrics.
    
    Analyzes prediction patterns and feature distributions over time.
    """
    try:
        # Get drift metrics from database
        drift_metrics = db_manager.get_model_drift_metrics(
            days=days,
            model_version=model_version
        )
        
        return {
            "model_version": model_version or "current",
            "analysis_period_days": days,
            "drift_metrics": drift_metrics,
            "drift_status": _assess_drift_status(drift_metrics),
            "recommendations": _generate_drift_recommendations(drift_metrics),
            "timestamp": get_utc_now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error retrieving drift metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Drift metrics error",
                "message": str(e)
            }
        )


def _generate_model_comparison_analysis(results: List[Dict], metrics: List[str]) -> Dict[str, Any]:
    """Generate analysis of model comparison results."""
    analysis = {
        "summary": "Model comparison analysis",
        "best_performing": {},
        "trends": {},
        "recommendations": []
    }
    
    # Find best performing model for each metric
    for metric in metrics:
        best_model = None
        best_value = None
        
        for result in results:
            perf_metrics = result.get('performance_metrics', {})
            value = perf_metrics.get(metric)
            
            if value is not None:
                if best_value is None or value > best_value:
                    best_value = value
                    best_model = result['model_version']
        
        if best_model:
            analysis['best_performing'][metric] = {
                'model_version': best_model,
                'value': best_value
            }
    
    return analysis


def _generate_model_recommendation(results: List[Dict]) -> Dict[str, Any]:
    """Generate model deployment recommendation."""
    if not results:
        return {"recommendation": "No models to compare"}
    
    # Simple recommendation based on most recent active model
    active_models = [r for r in results if r.get('status') == 'active']
    
    if active_models:
        recommended = active_models[0]
        return {
            "recommendation": "continue_current",
            "model_version": recommended['model_version'],
            "reason": "Current active model performing adequately"
        }
    else:
        latest = max(results, key=lambda x: x['created_at'])
        return {
            "recommendation": "activate_latest",
            "model_version": latest['model_version'],
            "reason": "Activate most recent model version"
        }


def _assess_drift_status(drift_metrics: Dict) -> str:
    """Assess overall drift status from metrics."""
    if not drift_metrics:
        return "insufficient_data"
    
    # Simple drift assessment - would implement proper statistical tests
    drift_score = drift_metrics.get('overall_drift_score', 0)
    
    if drift_score < 0.1:
        return "stable"
    elif drift_score < 0.3:
        return "minor_drift"
    elif drift_score < 0.5:
        return "moderate_drift"
    else:
        return "significant_drift"


def _generate_drift_recommendations(drift_metrics: Dict) -> List[str]:
    """Generate recommendations based on drift analysis."""
    recommendations = []
    
    drift_status = _assess_drift_status(drift_metrics)
    
    if drift_status == "significant_drift":
        recommendations.append("Consider retraining model with recent data")
        recommendations.append("Review feature importance and distributions")
    elif drift_status == "moderate_drift":
        recommendations.append("Monitor closely and prepare for retraining")
        recommendations.append("Analyze prediction accuracy trends")
    elif drift_status == "stable":
        recommendations.append("Model performance is stable")
        recommendations.append("Continue regular monitoring")
    
    return recommendations


@router.put("/{model_version}/deploy", response_model=Dict[str, Any])
async def deploy_model_to_production(model_version: str):
    """
    Deploy a specific model version to production.
    
    Sets the specified model version as active and reloads the model server.
    """
    try:
        model_record = db_manager.get_model_details(model_version)
        if not model_record:
            raise ModelNotFoundError(f"Model version {model_version} not found.")

        model_record.stage = "production"
        model_record.updated_at = get_utc_now()
        
        db_manager.update_model_registry(model_record)

        # Reload the model server to use the new production model
        get_model_server().reload_model()
        
        logger.info(f"Model {model_version} successfully deployed to production.")
        return {
            "message": "Model deployed to production successfully",
            "model_version": model_version,
            "timestamp": get_utc_now().isoformat()
        }
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error deploying model to production: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Failed to deploy model",
                "message": str(e),
                "timestamp": get_utc_now().isoformat()
            }
        )


@router.post("/retrain", response_model=Dict[str, Any])
async def trigger_model_retraining():
    """
    Trigger model retraining.
    
    This endpoint would typically trigger a background job to retrain the model.
    For now, it runs the retraining synchronously for simplicity.
    """
    try:
        # This would trigger a background job in a real system
        # For now, we'll run it synchronously for simplicity
        from models.model_training import ModelTrainingPipeline
        
        pipeline = ModelTrainingPipeline()
        new_model_version = pipeline.run_training_pipeline()
        
        return {
            "message": "Model retraining job started successfully.",
            "new_model_version": new_model_version,
            "timestamp": get_utc_now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to trigger model retraining: {e}")
        raise HTTPException(status_code=500, detail={
            "error": "Failed to start retraining job",
            "message": str(e),
            "timestamp": get_utc_now().isoformat()
        })