"""
Logging configuration for RiskFlow Credit Risk MLOps Pipeline.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

from .settings import get_settings


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """
    Set up application logging with both console and file handlers.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        log_format: Custom log format string
    
    Returns:
        Configured logger instance
    """
    settings = get_settings()
    
    # Use provided values or fall back to settings
    log_level = log_level or settings.log_level
    log_file = log_file or settings.log_file
    log_format = log_format or settings.log_format
    
    # Create logs directory if it doesn't exist
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    logger = logging.getLogger("riskflow")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    if log_format.lower() == 'json':
        # For JSON format, use a simple format and handle JSON in custom formatter
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=settings.log_max_size,
        backupCount=settings.log_backup_count,
        encoding="utf-8"
    )
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Set third-party loggers to WARNING to reduce noise
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(f"riskflow.{name}")


def log_function_call(func_name: str, **kwargs) -> None:
    """
    Log function call with parameters.
    
    Args:
        func_name: Name of the function being called
        **kwargs: Function parameters to log
    """
    logger = get_logger("function_calls")
    params = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
    logger.debug(f"Calling {func_name}({params})")


def log_model_metrics(
    model_name: str,
    metrics: dict,
    experiment_id: str,
    run_id: str
) -> None:
    """
    Log model training/validation metrics.
    
    Args:
        model_name: Name of the model
        metrics: Dictionary of metrics
        experiment_id: MLflow experiment ID
        run_id: MLflow run ID
    """
    logger = get_logger("model_metrics")
    metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
    logger.info(
        f"Model: {model_name} | Experiment: {experiment_id} | "
        f"Run: {run_id} | Metrics: {metrics_str}"
    )


def log_api_request(
    method: str,
    endpoint: str,
    status_code: int,
    response_time: float,
    user_id: Optional[str] = None
) -> None:
    """
    Log API request details.
    
    Args:
        method: HTTP method
        endpoint: API endpoint
        status_code: HTTP status code
        response_time: Response time in seconds
        user_id: Optional user identifier
    """
    logger = get_logger("api_requests")
    user_info = f" | User: {user_id}" if user_id else ""
    logger.info(
        f"{method} {endpoint} | Status: {status_code} | "
        f"Time: {response_time:.3f}s{user_info}"
    )


def log_prediction_request(
    model_name: str,
    input_features: dict,
    prediction: dict,
    confidence: float,
    request_id: str
) -> None:
    """
    Log prediction request and response.
    
    Args:
        model_name: Name of the model used
        input_features: Input features for prediction
        prediction: Model prediction result
        confidence: Prediction confidence score
        request_id: Unique request identifier
    """
    logger = get_logger("predictions")
    logger.info(
        f"Prediction | Request: {request_id} | Model: {model_name} | "
        f"Confidence: {confidence:.3f} | Features: {len(input_features)} | "
        f"Result: {prediction}"
    )


def log_model_drift(
    model_name: str,
    drift_score: float,
    threshold: float,
    feature_drifts: dict
) -> None:
    """
    Log model drift detection results.
    
    Args:
        model_name: Name of the model
        drift_score: Overall drift score
        threshold: Drift threshold
        feature_drifts: Per-feature drift scores
    """
    logger = get_logger("model_drift")
    drift_status = "ALERT" if drift_score > threshold else "OK"
    top_drifting_features = sorted(
        feature_drifts.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:3]
    
    logger.warning(
        f"Drift Detection | Model: {model_name} | Status: {drift_status} | "
        f"Score: {drift_score:.4f} (threshold: {threshold:.4f}) | "
        f"Top drifting features: {top_drifting_features}"
    )


def log_error(
    error: Exception,
    context: str,
    additional_info: Optional[dict] = None
) -> None:
    """
    Log error with context and additional information.
    
    Args:
        error: Exception instance
        context: Context where error occurred
        additional_info: Additional information to log
    """
    logger = get_logger("errors")
    additional = ""
    if additional_info:
        additional = f" | Additional info: {additional_info}"
    
    logger.error(
        f"Error in {context}: {str(error)}{additional}",
        exc_info=True
    )


# Performance monitoring decorators
def log_execution_time(func):
    """Decorator to log function execution time."""
    import time
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger = get_logger("performance")
            logger.debug(f"{func.__name__} executed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger = get_logger("performance")
            logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {str(e)}")
            raise
    
    return wrapper


# Initialize logging on module import
_logger = setup_logging()
_logger.info("RiskFlow logging system initialized")