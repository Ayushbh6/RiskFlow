"""
Custom exceptions for RiskFlow Credit Risk MLOps Pipeline.
Provides structured error handling throughout the application.
"""

from typing import Optional, Dict, Any


class RiskFlowException(Exception):
    """
    Base exception class for RiskFlow application.
    All custom exceptions should inherit from this class.
    """
    
    def __init__(
        self,
        message: str,
        error_type: str = "GENERAL_ERROR",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize RiskFlow exception.
        
        Args:
            message: Human-readable error message
            error_type: Machine-readable error type
            status_code: HTTP status code for API responses
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.status_code = status_code
        self.details = details or {}


class DataSourceException(RiskFlowException):
    """Exception raised when data source operations fail."""
    
    def __init__(self, message: str, source: str, details: Optional[Dict[str, Any]] = None):
        base_details = details or {}
        base_details["data_source"] = source
        super().__init__(
            message=message,
            error_type="DATA_SOURCE_ERROR",
            status_code=503,
            details=base_details
        )


class ModelException(RiskFlowException):
    """Exception raised when model operations fail."""
    
    def __init__(self, message: str, model_name: str = "unknown", details: Optional[Dict[str, Any]] = None):
        base_details = details or {}
        base_details["model_name"] = model_name
        super().__init__(
            message=message,
            error_type="MODEL_ERROR",
            status_code=500,
            details=base_details
        )


class ValidationException(RiskFlowException):
    """Exception raised when data validation fails."""
    
    def __init__(self, message: str, validation_errors: Optional[list] = None):
        super().__init__(
            message=message,
            error_type="VALIDATION_ERROR",
            status_code=422,
            details={"validation_errors": validation_errors or []}
        )


class PredictionException(RiskFlowException):
    """Exception raised when prediction operations fail."""
    
    def __init__(self, message: str, request_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        base_details = details or {}
        base_details["request_id"] = request_id
        super().__init__(
            message=message,
            error_type="PREDICTION_ERROR",
            status_code=500,
            details=base_details
        )


class DataIngestionError(RiskFlowException):
    """Exception raised when data ingestion fails."""
    
    def __init__(self, message: str, source: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        base_details = details or {}
        base_details["data_source"] = source
        super().__init__(
            message=message,
            error_type="DATA_INGESTION_ERROR",
            status_code=503,
            details=base_details
        )


class ModelError(RiskFlowException):
    """Exception raised when model operations fail."""
    
    def __init__(self, message: str, model_name: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        base_details = details or {}
        base_details["model_name"] = model_name
        super().__init__(
            message=message,
            error_type="MODEL_ERROR",
            status_code=500,
            details=base_details
        )


class APIError(RiskFlowException):
    """Exception raised for API-specific errors."""
    
    def __init__(self, message: str, endpoint: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        base_details = details or {}
        base_details["endpoint"] = endpoint
        super().__init__(
            message=message,
            error_type="API_ERROR",
            status_code=500,
            details=base_details
        )


class ModelNotFoundError(RiskFlowException):
    """Exception raised when a model is not found."""
    
    def __init__(self, message: str, model_name: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        base_details = details or {}
        base_details["model_name"] = model_name
        super().__init__(
            message=message,
            error_type="MODEL_NOT_FOUND_ERROR",
            status_code=404,
            details=base_details
        )


class InvalidInputError(RiskFlowException):
    """Exception raised for invalid input data."""
    
    def __init__(self, message: str, input_data: Optional[Dict] = None, details: Optional[Dict[str, Any]] = None):
        base_details = details or {}
        base_details["input_data"] = input_data
        super().__init__(
            message=message,
            error_type="INVALID_INPUT_ERROR",
            status_code=400,
            details=base_details
        )


class DataTransformationError(RiskFlowException):
    """Exception raised for data transformation errors."""
    
    def __init__(self, message: str, field: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        base_details = details or {}
        base_details["field"] = field
        super().__init__(
            message=message,
            error_type="DATA_TRANSFORMATION_ERROR",
            status_code=422,
            details=base_details
        )


class ModelServingError(RiskFlowException):
    """Exception raised for model serving errors."""
    
    def __init__(self, message: str, model_name: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        base_details = details or {}
        base_details["model_name"] = model_name
        super().__init__(
            message=message,
            error_type="MODEL_SERVING_ERROR",
            status_code=500,
            details=base_details
        )


class ModelTrainingError(RiskFlowException):
    """Exception raised for model training errors."""
    
    def __init__(self, message: str, model_name: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        base_details = details or {}
        base_details["model_name"] = model_name
        super().__init__(
            message=message,
            error_type="MODEL_TRAINING_ERROR",
            status_code=500,
            details=base_details
        )


class ModelValidationError(RiskFlowException):
    """Exception raised for model validation errors."""
    
    def __init__(self, message: str, model_name: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        base_details = details or {}
        base_details["model_name"] = model_name
        super().__init__(
            message=message,
            error_type="MODEL_VALIDATION_ERROR",
            status_code=400,
            details=base_details
        )


class DatabaseException(RiskFlowException):
    """Exception raised when database operations fail."""
    
    def __init__(self, message: str, operation: str = "unknown", details: Optional[Dict[str, Any]] = None):
        base_details = details or {}
        base_details["operation"] = operation
        super().__init__(
            message=message,
            error_type="DATABASE_ERROR",
            status_code=500,
            details=base_details
        )


class ConfigurationException(RiskFlowException):
    """Exception raised when configuration is invalid."""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(
            message=message,
            error_type="CONFIGURATION_ERROR",
            status_code=500,
            details={"config_key": config_key}
        )


class AuthenticationException(RiskFlowException):
    """Exception raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(
            message=message,
            error_type="AUTHENTICATION_ERROR",
            status_code=401
        )


class AuthorizationException(RiskFlowException):
    """Exception raised when authorization fails."""
    
    def __init__(self, message: str = "Authorization failed", required_permission: Optional[str] = None):
        super().__init__(
            message=message,
            error_type="AUTHORIZATION_ERROR",
            status_code=403,
            details={"required_permission": required_permission}
        )


class RateLimitException(RiskFlowException):
    """Exception raised when rate limits are exceeded."""
    
    def __init__(self, message: str, retry_after: Optional[int] = None):
        super().__init__(
            message=message,
            error_type="RATE_LIMIT_ERROR",
            status_code=429,
            details={"retry_after": retry_after}
        )


class FeatureEngineeringException(RiskFlowException):
    """Exception raised when feature engineering fails."""
    
    def __init__(self, message: str, feature_name: Optional[str] = None):
        super().__init__(
            message=message,
            error_type="FEATURE_ENGINEERING_ERROR",
            status_code=500,
            details={"feature_name": feature_name}
        )


class MLPipelineException(RiskFlowException):
    """Exception raised when ML pipeline operations fail."""
    
    def __init__(self, message: str, pipeline_stage: str = "unknown", details: Optional[Dict[str, Any]] = None):
        base_details = details or {}
        base_details["pipeline_stage"] = pipeline_stage
        super().__init__(
            message=message,
            error_type="ML_PIPELINE_ERROR",
            status_code=500,
            details=base_details
        )


class LLMError(RiskFlowException):
    """Exception raised when LLM operations fail."""
    
    def __init__(self, message: str, provider: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        base_details = details or {}
        base_details["provider"] = provider
        super().__init__(
            message=message,
            error_type="LLM_ERROR",
            status_code=503,
            details=base_details
        )


class DocumentationError(RiskFlowException):
    """Exception raised when documentation generation fails."""
    
    def __init__(self, message: str, doc_type: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        base_details = details or {}
        base_details["documentation_type"] = doc_type
        super().__init__(
            message=message,
            error_type="DOCUMENTATION_ERROR",
            status_code=500,
            details=base_details
        )


# Error handling utilities
def handle_exception(e: Exception) -> RiskFlowException:
    """
    Convert generic exceptions to RiskFlow exceptions.
    
    Args:
        e: The original exception
        
    Returns:
        RiskFlowException with appropriate error type
    """
    if isinstance(e, RiskFlowException):
        return e
    
    # Map common exception types
    if isinstance(e, ValueError):
        return ValidationException(str(e))
    elif isinstance(e, KeyError):
        return ConfigurationException(f"Missing required key: {str(e)}")
    elif isinstance(e, ConnectionError):
        return DataSourceException(str(e), "connection")
    elif isinstance(e, TimeoutError):
        return DataSourceException(str(e), "timeout")
    else:
        return RiskFlowException(
            message=f"Unexpected error: {str(e)}",
            error_type="UNEXPECTED_ERROR",
            details={"original_exception": type(e).__name__}
        )


def format_error_response(exception: RiskFlowException) -> Dict[str, Any]:
    """
    Format exception as API error response.
    
    Args:
        exception: RiskFlow exception to format
        
    Returns:
        Dictionary formatted for API response
    """
    return {
        "error": exception.error_type,
        "message": exception.message,
        "details": exception.details,
        "status_code": exception.status_code
    }