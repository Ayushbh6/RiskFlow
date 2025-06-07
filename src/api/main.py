"""
FastAPI main application for RiskFlow Credit Risk MLOps Pipeline.
Provides REST API endpoints for credit risk scoring and model management.
"""

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import time
import uuid
from typing import Dict, Any
from contextlib import asynccontextmanager

from api.routes import health, predictions, models
from api.middleware.logging import LoggingMiddleware
from api.middleware.rate_limiting import RateLimitMiddleware
from config.settings import get_settings
from config.logging_config import get_logger
from utils.database import initialize_database
from utils.exceptions import RiskFlowException

logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown tasks.
    """
    # Startup
    logger.info("Starting RiskFlow API server")
    
    # Store application start time globally
    app.state.start_time = time.time()
    
    # Initialize database
    try:
        initialize_database()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        raise
    
    # Additional startup tasks
    logger.info(f"RiskFlow API v{settings.app_version} started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down RiskFlow API server")


# Create FastAPI application
app = FastAPI(
    title="RiskFlow Credit Risk API",
    description="Production-ready MLOps API for credit risk scoring and model management",
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    openapi_url="/openapi.json" if settings.debug else None,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Add trusted host middleware for security (disabled for development)
# if not settings.debug:
#     app.add_middleware(
#         TrustedHostMiddleware,
#         allowed_hosts=["localhost", "127.0.0.1", "*.riskflow.app", "testserver"]
#     )

# Add custom middleware
app.add_middleware(LoggingMiddleware)
app.add_middleware(RateLimitMiddleware)


# Global exception handler
@app.exception_handler(RiskFlowException)
async def riskflow_exception_handler(request: Request, exc: RiskFlowException):
    """
    Handle custom RiskFlow exceptions.
    """
    logger.error(f"RiskFlow exception: {exc.message} - {exc.details}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error_type,
            "message": exc.message,
            "details": exc.details,
            "request_id": str(uuid.uuid4()),
            "timestamp": time.time()
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Handle FastAPI HTTP exceptions.
    """
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP_ERROR",
            "message": exc.detail,
            "status_code": exc.status_code,
            "request_id": str(uuid.uuid4()),
            "timestamp": time.time()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Handle unexpected exceptions.
    """
    logger.error(f"Unexpected exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred. Please try again later.",
            "request_id": str(uuid.uuid4()),
            "timestamp": time.time()
        }
    )


# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "service": "RiskFlow Credit Risk API",
        "version": settings.app_version,
        "status": "operational",
        "documentation": "/docs",
        "health_check": "/health",
        "timestamp": time.time()
    }


# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(predictions.router, prefix="/api/v1/predictions", tags=["predictions"])
app.include_router(models.router, prefix="/api/v1/models", tags=["models"])


# Custom OpenAPI schema
def custom_openapi():
    """
    Generate custom OpenAPI schema.
    """
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="RiskFlow Credit Risk API",
        version=settings.app_version,
        description="""
        ## Production-Ready MLOps API for Credit Risk Scoring
        
        ### Key Features:
        - **Real-time Credit Risk Scoring**: PD, LGD, and Expected Loss calculations
        - **Model Management**: Version control and A/B testing capabilities  
        - **Performance Monitoring**: Real-time model performance tracking
        - **Production Ready**: Rate limiting, caching, and comprehensive error handling
        
        ### Authentication:
        API key required for all endpoints except health checks.
        Include `X-API-Key` header with your API key.
        
        ### Rate Limits:
        - Standard tier: 100 requests per minute
        - Premium tier: 1000 requests per minute
        
        ### Response Times:
        - Prediction endpoints: <100ms target
        - Model operations: <500ms target
        """,
        routes=app.routes,
    )
    
    # Add custom info
    openapi_schema["info"]["x-logo"] = {
        "url": "https://riskflow.app/logo.png"
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting RiskFlow API server on {settings.api_host}:{settings.api_port}")
    
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        workers=settings.api_workers if not settings.api_reload else 1,
        log_level="info",
        access_log=True
    )
