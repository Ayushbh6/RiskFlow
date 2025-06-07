"""
Prediction endpoints for RiskFlow Credit Risk API.
Provides real-time credit risk scoring and batch predictions.
"""

from fastapi import APIRouter, HTTPException, Depends, status, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
import time
import logging

from ..schemas.prediction import (
    CreditRiskRequest, 
    CreditRiskResponse, 
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionStatus
)
from ...models.model_serving import get_model_server, predict_credit_risk
from ...data.validators import validate_real_time_data
from ...data.preprocessing import CreditRiskFeatureEngineer
from ...utils.database import db_manager
from ...config.settings import get_settings
from ...config.logging_config import get_logger
from ...utils.exceptions import APIError, ModelNotFoundError, InvalidInputError
from ...utils.helpers import get_utc_now

logger = get_logger(__name__)
settings = get_settings()
router = APIRouter()


@router.post("/single", response_model=CreditRiskResponse)
async def predict_single_application(
    request: CreditRiskRequest,
    background_tasks: BackgroundTasks
):
    """
    Predict credit risk for a single loan application.
    
    Returns probability of default, loss given default, expected loss,
    and risk rating with detailed analysis.
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        logger.info(f"Processing credit risk prediction for request {request_id}")
        
        # Convert request to dict for processing
        credit_data = request.dict()
        credit_data['request_id'] = request_id
        
        # Validate input data
        validation_result = validate_real_time_data([credit_data], 'credit')
        if not validation_result['is_valid']:
            logger.warning(f"Invalid credit data for request {request_id}: {validation_result['errors']}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": "Invalid credit application data",
                    "validation_errors": validation_result['errors'],
                    "request_id": request_id
                }
            )
        
        # Get model server and make prediction
        model_server = get_model_server()
        prediction_result = model_server.predict(
            credit_data,
            use_cache=True,
            log_prediction=True
        )
        
        # Check if prediction was successful
        if not prediction_result.get('success', False):
            error_msg = prediction_result.get('error', 'Unknown prediction error')
            logger.error(f"Prediction failed for request {request_id}: {error_msg}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "Prediction service error",
                    "message": error_msg,
                    "request_id": request_id
                }
            )
        
        # Extract prediction data
        predictions = prediction_result['predictions']
        response_time_ms = (time.time() - start_time) * 1000
        
        # Create response
        response = CreditRiskResponse(
            request_id=request_id,
            success=True,
            predictions={
                "probability_of_default": predictions['probability_of_default'],
                "loss_given_default": predictions['loss_given_default'],
                "expected_loss": predictions['expected_loss'],
                "risk_rating": predictions['risk_rating'],
                "risk_category": predictions['risk_category'],
                "decision": predictions['decision']
            },
            confidence_scores=prediction_result.get('confidence_scores', {}),
            economic_context=prediction_result.get('economic_context', {}),
            model_version=prediction_result.get('model_version'),
            response_time_ms=response_time_ms,
            timestamp=get_utc_now().isoformat()
        )
        
        # Log successful prediction
        background_tasks.add_task(
            _log_api_usage,
            request_id,
            "single_prediction",
            True,
            response_time_ms
        )
        
        logger.info(f"Credit risk prediction completed for request {request_id} in {response_time_ms:.2f}ms")
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in credit risk prediction for request {request_id}: {str(e)}")
        
        # Log failed prediction
        background_tasks.add_task(
            _log_api_usage,
            request_id,
            "single_prediction",
            False,
            (time.time() - start_time) * 1000,
            str(e)
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Internal server error",
                "message": "An unexpected error occurred during prediction",
                "request_id": request_id,
                "timestamp": get_utc_now().isoformat()
            }
        )


@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch_applications(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks
):
    """
    Predict credit risk for multiple loan applications.
    
    Processes applications in batches for optimal performance.
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        logger.info(f"Processing batch prediction for {len(request.applications)} applications, request {request_id}")
        
        # Validate batch size
        if len(request.applications) > settings.max_batch_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail={
                    "error": "Batch size too large",
                    "message": f"Maximum batch size is {settings.max_batch_size}",
                    "request_id": request_id
                }
            )
        
        # Convert applications to dict format
        credit_applications = [app.dict() for app in request.applications]
        
        # Add request IDs to each application
        for i, app in enumerate(credit_applications):
            app['batch_request_id'] = request_id
            app['application_index'] = i
        
        # Validate all applications
        validation_result = validate_real_time_data(credit_applications, 'credit')
        if not validation_result['is_valid']:
            logger.warning(f"Invalid batch data for request {request_id}: {validation_result['errors']}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": "Invalid batch application data",
                    "validation_errors": validation_result['errors'],
                    "request_id": request_id
                }
            )
        
        # Get model server and make batch predictions
        model_server = get_model_server()
        batch_results = model_server.batch_predict(
            credit_applications,
            max_batch_size=request.batch_size or 50
        )
        
        # Process results
        successful_predictions = []
        failed_predictions = []
        
        for i, result in enumerate(batch_results):
            if result.get('success', False):
                successful_predictions.append({
                    "application_index": i,
                    "predictions": result['predictions'],
                    "confidence_scores": result.get('confidence_scores', {}),
                    "response_time_ms": result.get('response_time_ms', 0)
                })
            else:
                failed_predictions.append({
                    "application_index": i,
                    "error": result.get('error', 'Unknown error'),
                    "error_details": result.get('error_details', '')
                })
        
        response_time_ms = (time.time() - start_time) * 1000
        
        # Create response
        response = BatchPredictionResponse(
            request_id=request_id,
            success=len(failed_predictions) == 0,
            total_applications=len(request.applications),
            successful_predictions=len(successful_predictions),
            failed_predictions=len(failed_predictions),
            results=successful_predictions,
            errors=failed_predictions,
            response_time_ms=response_time_ms,
            timestamp=get_utc_now().isoformat()
        )
        
        # Log batch prediction
        background_tasks.add_task(
            _log_api_usage,
            request_id,
            "batch_prediction",
            response.success,
            response_time_ms,
            None,
            len(request.applications)
        )
        
        logger.info(
            f"Batch prediction completed for request {request_id}: "
            f"{successful_predictions}/{len(request.applications)} successful in {response_time_ms:.2f}ms"
        )
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in batch prediction for request {request_id}: {str(e)}")
        
        # Log failed batch prediction
        background_tasks.add_task(
            _log_api_usage,
            request_id,
            "batch_prediction",
            False,
            (time.time() - start_time) * 1000,
            str(e),
            len(request.applications) if hasattr(request, 'applications') else 0
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Internal server error",
                "message": "An unexpected error occurred during batch prediction",
                "request_id": request_id,
                "timestamp": get_utc_now().isoformat()
            }
        )


@router.get("/status/{request_id}", response_model=Dict[str, Any])
async def get_prediction_status(request_id: str):
    """
    Get the status of a prediction request.
    
    Useful for tracking long-running batch predictions.
    """
    try:
        # Query prediction log for status
        prediction_log = db_manager.get_prediction_log(request_id)
        
        if not prediction_log:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "Prediction not found",
                    "message": f"No prediction found with request_id: {request_id}",
                    "request_id": request_id
                }
            )
        
        return {
            "request_id": request_id,
            "status": "completed",  # For now, all predictions are synchronous
            "timestamp": prediction_log.get('timestamp'),
            "response_time_ms": prediction_log.get('response_time_ms'),
            "model_version": prediction_log.get('model_version'),
            "success": prediction_log.get('success', False)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving prediction status for {request_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Status retrieval error",
                "message": str(e),
                "request_id": request_id
            }
        )


@router.get("/history", response_model=Dict[str, Any])
async def get_prediction_history(
    limit: int = 100,
    offset: int = 0,
    customer_id: Optional[str] = None
):
    """
    Get prediction history with optional filtering.
    
    Returns recent predictions with pagination support.
    """
    try:
        # Validate parameters
        if limit > 1000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Limit cannot exceed 1000"
            )
        
        # Get prediction history from database
        history = db_manager.get_prediction_history(
            limit=limit,
            offset=offset,
            customer_id=customer_id
        )
        
        return {
            "predictions": history,
            "pagination": {
                "limit": limit,
                "offset": offset,
                "total_returned": len(history),
                "has_more": len(history) == limit
            },
            "filters": {
                "customer_id": customer_id
            },
            "timestamp": get_utc_now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving prediction history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Failed to retrieve prediction history",
                "message": str(e),
                "timestamp": get_utc_now().isoformat()
            }
        )


async def _log_api_usage(
    request_id: str,
    endpoint_type: str,
    success: bool,
    response_time_ms: float,
    error_message: Optional[str] = None,
    batch_size: Optional[int] = None
):
    """
    Background task to log API usage statistics.
    """
    try:
        db_manager.log_api_usage(
            request_id=request_id,
            endpoint=endpoint_type,
            success=success,
            response_time_ms=response_time_ms,
            error_message=error_message,
            batch_size=batch_size
        )
    except Exception as e:
        logger.error(f"Failed to log API usage for request {request_id}: {str(e)}")


# Warm up model server on module load
try:
    from ...models.model_serving import warm_up_model_server
    warm_up_model_server()
except Exception as e:
    logger.warning(f"Model server warmup failed: {str(e)}")