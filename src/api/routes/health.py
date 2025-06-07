"""
Health check endpoints for RiskFlow Credit Risk API.
Provides comprehensive system health monitoring.
"""

from fastapi import APIRouter, HTTPException, status, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any
from datetime import datetime
import time
import asyncio

from data.data_sources import get_real_market_data
from utils.database import db_manager
from models.model_serving import get_model_server
from config.settings import get_settings
from config.logging_config import get_logger
from utils.helpers import get_utc_now

logger = get_logger(__name__)
settings = get_settings()
router = APIRouter()


@router.get("/", response_model=Dict[str, Any])
async def basic_health_check(request: Request):
    """
    Basic health check endpoint.
    Returns simple status and timestamp.
    """
    # Get start time from app state
    start_time = getattr(request.app.state, 'start_time', time.time())
    uptime_seconds = time.time() - start_time
    
    return {
        "status": "healthy",
        "service": "RiskFlow Credit Risk API",
        "version": settings.app_version,
        "timestamp": get_utc_now().isoformat(),
        "uptime_seconds": uptime_seconds
    }


@router.get("/detailed", response_model=Dict[str, Any])
async def detailed_health_check():
    """
    Detailed health check with component status.
    Checks database, data sources, and model availability.
    """
    start_time = time.time()
    health_status = {
        "status": "unknown",
        "service": "RiskFlow Credit Risk API",
        "version": settings.app_version,
        "timestamp": get_utc_now().isoformat(),
        "components": {},
        "overall_health": True
    }
    
    # Check database connectivity
    try:
        db_healthy = db_manager.health_check()
        health_status["components"]["database"] = {
            "status": "healthy" if db_healthy else "unhealthy",
            "message": "Database connection successful" if db_healthy else "Database connection failed",
            "response_time_ms": (time.time() - start_time) * 1000
        }
        if not db_healthy:
            health_status["overall_health"] = False
    except Exception as e:
        health_status["components"]["database"] = {
            "status": "unhealthy",
            "message": f"Database error: {str(e)}",
            "response_time_ms": (time.time() - start_time) * 1000
        }
        health_status["overall_health"] = False
    
    # Check real data sources
    data_check_start = time.time()
    try:
        market_data = get_real_market_data()
        if market_data is not None:
            health_status["components"]["data_sources"] = {
                "status": "healthy",
                "message": "Real market data sources available",
                "data_quality": market_data.get("data_quality", "unknown"),
                "response_time_ms": (time.time() - data_check_start) * 1000
            }
        else:
            health_status["components"]["data_sources"] = {
                "status": "degraded",
                "message": "Real market data sources unavailable",
                "response_time_ms": (time.time() - data_check_start) * 1000
            }
    except Exception as e:
        health_status["components"]["data_sources"] = {
            "status": "unhealthy",
            "message": f"Data source error: {str(e)}",
            "response_time_ms": (time.time() - data_check_start) * 1000
        }
    
    # Check model server
    model_check_start = time.time()
    try:
        model_server = get_model_server()
        model_info = model_server.get_model_info()
        
        if model_info.get("is_loaded"):
            health_status["components"]["model_server"] = {
                "status": "healthy",
                "message": "Model server operational",
                "model_version": model_info.get("model_version"),
                "feature_count": model_info.get("feature_count"),
                "response_time_ms": (time.time() - model_check_start) * 1000
            }
        else:
            health_status["components"]["model_server"] = {
                "status": "degraded",
                "message": "Model server running but no model loaded",
                "response_time_ms": (time.time() - model_check_start) * 1000
            }
    except Exception as e:
        health_status["components"]["model_server"] = {
            "status": "unhealthy",
            "message": f"Model server error: {str(e)}",
            "response_time_ms": (time.time() - model_check_start) * 1000
        }
        health_status["overall_health"] = False
    
    # Determine overall status
    component_statuses = [comp["status"] for comp in health_status["components"].values()]
    if all(status == "healthy" for status in component_statuses):
        health_status["status"] = "healthy"
    elif any(status == "unhealthy" for status in component_statuses):
        health_status["status"] = "unhealthy"
        health_status["overall_health"] = False
    else:
        health_status["status"] = "degraded"
    
    health_status["total_response_time_ms"] = (time.time() - start_time) * 1000
    
    # Return appropriate HTTP status
    if health_status["status"] == "unhealthy":
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=health_status
        )
    elif health_status["status"] == "degraded":
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=health_status
        )
    else:
        return health_status


@router.get("/readiness", response_model=Dict[str, Any])
async def readiness_check():
    """
    Kubernetes-style readiness probe.
    Checks if service is ready to receive traffic.
    """
    try:
        # Check critical components only
        db_ready = db_manager.health_check()
        
        if db_ready:
            return {
                "status": "ready",
                "message": "Service ready to receive traffic",
                "timestamp": get_utc_now().isoformat()
            }
        else:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "not_ready",
                    "message": "Database not available",
                    "timestamp": get_utc_now().isoformat()
                }
            )
    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "not_ready",
                "message": f"Readiness check error: {str(e)}",
                "timestamp": get_utc_now().isoformat()
            }
        )


@router.get("/liveness", response_model=Dict[str, Any])
async def liveness_check():
    """
    Kubernetes-style liveness probe.
    Checks if service is alive and should not be restarted.
    """
    try:
        # Simple check - if we can respond, we're alive
        return {
            "status": "alive",
            "message": "Service is alive",
            "timestamp": get_utc_now().isoformat(),
            "process_id": getattr(liveness_check, '_process_id', 'unknown')
        }
    except Exception as e:
        logger.error(f"Liveness check failed: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "dead",
                "message": f"Liveness check error: {str(e)}",
                "timestamp": get_utc_now().isoformat()
            }
        )


@router.get("/metrics", response_model=Dict[str, Any])
async def health_metrics(request: Request):
    """
    Health metrics endpoint for monitoring systems.
    Provides detailed metrics about system performance.
    """
    try:
        # Get start time from app state
        start_time = getattr(request.app.state, 'start_time', time.time())
        uptime_seconds = time.time() - start_time
        
        metrics = {
            "timestamp": get_utc_now().isoformat(),
            "service_metrics": {
                "requests_total": getattr(health_metrics, '_request_count', 0),
                "uptime_seconds": uptime_seconds,
                "memory_usage_mb": "not_implemented",  # Could use psutil
                "cpu_usage_percent": "not_implemented"  # Could use psutil
            },
            "database_metrics": {},
            "model_metrics": {},
            "data_source_metrics": {}
        }
        
        # Database metrics
        try:
            db_stats = db_manager.get_database_stats()
            metrics["database_metrics"] = {
                "connection_status": "healthy" if db_manager.health_check() else "unhealthy",
                "total_credit_records": db_stats.get("credit_data_count", 0),
                "total_predictions": db_stats.get("prediction_count", 0),
                "last_update": db_stats.get("last_update", "unknown")
            }
        except Exception as e:
            metrics["database_metrics"]["error"] = str(e)
        
        # Model metrics
        try:
            model_server = get_model_server()
            model_info = model_server.get_model_info()
            metrics["model_metrics"] = {
                "model_loaded": model_info.get("is_loaded", False),
                "model_version": model_info.get("model_version", "unknown"),
                "cache_size": model_info.get("cache_size", 0),
                "feature_count": model_info.get("feature_count", 0)
            }
        except Exception as e:
            metrics["model_metrics"]["error"] = str(e)
        
        # Data source metrics
        try:
            market_data = get_real_market_data()
            if market_data:
                metrics["data_source_metrics"] = {
                    "market_data_available": True,
                    "data_quality": market_data.get("data_quality", "unknown"),
                    "last_update": market_data.get("timestamp", "unknown"),
                    "economic_factors_count": len(market_data.get("economic_factors", {}))
                }
            else:
                metrics["data_source_metrics"] = {
                    "market_data_available": False,
                    "message": "Real market data sources unavailable"
                }
        except Exception as e:
            metrics["data_source_metrics"]["error"] = str(e)
        
        # Increment request counter
        health_metrics._request_count = getattr(health_metrics, '_request_count', 0) + 1
        
        return metrics
        
    except Exception as e:
        logger.error(f"Health metrics failed: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Failed to generate health metrics",
                "message": str(e),
                "timestamp": get_utc_now().isoformat()
            }
        )


# Initialize start time for uptime calculation
basic_health_check._start_time = time.time()
health_metrics._start_time = time.time()
health_metrics._request_count = 0


@router.get("/dependencies", response_model=Dict[str, Any])
async def check_dependencies():
    """
    Check the health of external dependencies.
    """
    dependencies = {
        "database": await check_database_health(),
        "data_source": await check_data_source_health(),
        "model_server": await check_model_server_health()
    }
    
    overall_status = "healthy" if all(d['status'] == 'healthy' for d in dependencies.values()) else "degraded"
    
    return {
        "overall_status": overall_status,
        "timestamp": get_utc_now().isoformat(),
        "dependencies": dependencies
    }


async def check_database_health() -> Dict[str, Any]:
    """
    Check the health of the database.
    """
    try:
        db_healthy = db_manager.health_check()
        return {
            "status": "healthy" if db_healthy else "unhealthy",
            "message": "Database connection successful" if db_healthy else "Database connection failed",
            "timestamp": get_utc_now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Database error: {str(e)}",
            "timestamp": get_utc_now().isoformat()
        }


async def check_data_source_health() -> Dict[str, Any]:
    """
    Check the health of data sources.
    """
    try:
        # This should test connectivity to FRED, Treasury, etc.
        market_data = get_real_market_data()
        
        if market_data:
            status = "healthy"
            message = "Data sources are available and responding"
            details = {
                "economic_factors_count": len(market_data.get("economic_factors", {})),
                "credit_spreads_available": "credit_spreads" in market_data,
                "data_quality": market_data.get("data_quality", "unknown"),
                "last_update": market_data.get("timestamp")
            }
        else:
            status = "unhealthy"
            message = "One or more data sources are unavailable"
            details = {}
            
    except Exception as e:
        status = "unhealthy"
        message = f"Data source check failed: {str(e)}"
        details = {"error": str(e)}
        
    return {
        "status": status,
        "message": message,
        "details": details,
        "timestamp": get_utc_now().isoformat()
    }


async def check_model_server_health() -> Dict[str, Any]:
    """
    Check the health of the model server.
    """
    try:
        server = get_model_server()
        info = server.get_model_info()
        
        if info.get('is_loaded'):
            status = "healthy"
            message = "Model server is active and model is loaded"
            details = {
                "model_name": info.get("model_name"),
                "model_version": info.get("model_version"),
                "cache_size": info.get("cache_size")
            }
        else:
            status = "unhealthy"
            message = "Model server is active, but no model is loaded"
            details = info
            
    except Exception as e:
        status = "unhealthy"
        message = f"Model server check failed: {str(e)}"
        details = {"error": str(e)}
        
    return {
        "status": status,
        "message": message,
        "details": details,
        "timestamp": get_utc_now().isoformat()
    }


@router.get("/system", response_model=Dict[str, Any])
async def system_health_report():
    """
    Provides a comprehensive system health report.
    """
    report = {
        "service_info": {
            "name": "RiskFlow API",
            "version": settings.app_version,
            "environment": settings.environment,
            "timestamp": get_utc_now().isoformat()
        },
        "dependencies": await check_dependencies()
    }
    
    return report


# Duplicate function removed - functionality merged into the first health_metrics function above

# Initialize start time and request counter
health_metrics._start_time = time.time()
health_metrics._request_count = 0