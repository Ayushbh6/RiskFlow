"""
Pydantic schemas for prediction request/response models.
Defines data validation and serialization for credit risk API.
"""

from pydantic import BaseModel, Field, validator, ConfigDict
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum


class HomeOwnershipType(str, Enum):
    """Valid home ownership types."""
    RENT = "RENT"
    OWN = "OWN"
    MORTGAGE = "MORTGAGE"
    OTHER = "OTHER"


class LoanPurpose(str, Enum):
    """Valid loan purposes."""
    DEBT_CONSOLIDATION = "debt_consolidation"
    CREDIT_CARD = "credit_card"
    HOME_IMPROVEMENT = "home_improvement"
    MAJOR_PURCHASE = "major_purchase"
    MEDICAL = "medical"
    SMALL_BUSINESS = "small_business"
    VACATION = "vacation"
    WEDDING = "wedding"
    OTHER = "other"


class RiskCategory(str, Enum):
    """Risk category classifications."""
    VERY_LOW_RISK = "Very Low Risk"
    LOW_RISK = "Low Risk"
    LOW_MEDIUM_RISK = "Low-Medium Risk"
    MEDIUM_RISK = "Medium Risk"
    MEDIUM_HIGH_RISK = "Medium-High Risk"
    HIGH_RISK = "High Risk"
    VERY_HIGH_RISK = "Very High Risk"
    EXTREME_RISK = "Extreme Risk"


class DecisionType(str, Enum):
    """Lending decision types."""
    APPROVE = "APPROVE"
    APPROVE_WITH_CONDITIONS = "APPROVE_WITH_CONDITIONS"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    DECLINE = "DECLINE"


class PredictionStatus(str, Enum):
    """Prediction processing status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class CreditRiskRequest(BaseModel):
    """
    Credit risk prediction request schema.
    Validates loan application data for risk assessment.
    """
    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "customer_id": "CUST_12345",
                "loan_amount": 25000.0,
                "loan_term": 36,
                "interest_rate": 0.125,
                "income": 65000.0,
                "debt_to_income": 0.35,
                "credit_score": 720,
                "employment_length": 5,
                "home_ownership": "RENT",
                "loan_purpose": "debt_consolidation",
                "previous_defaults": 0,
                "open_credit_lines": 8,
                "total_credit_lines": 12
            }
        }
    )

    customer_id: str = Field(..., description="Unique customer identifier", min_length=1, max_length=100)
    loan_amount: float = Field(..., description="Requested loan amount in USD", gt=0, le=1000000)
    loan_term: int = Field(..., description="Loan term in months", ge=12, le=360)
    interest_rate: float = Field(..., description="Annual interest rate as decimal", ge=0.01, le=0.50)
    income: float = Field(..., description="Annual income in USD", gt=0, le=10000000)
    debt_to_income: float = Field(..., description="Debt-to-income ratio as decimal", ge=0, le=1.0)
    credit_score: int = Field(..., description="FICO credit score", ge=300, le=850)
    employment_length: int = Field(..., description="Employment length in years", ge=0, le=50)
    home_ownership: HomeOwnershipType = Field(..., description="Home ownership status")
    loan_purpose: LoanPurpose = Field(..., description="Purpose of the loan")
    
    # Optional fields for enhanced risk assessment
    annual_income: Optional[float] = Field(None, description="Alternative income field", gt=0)
    previous_defaults: Optional[int] = Field(0, description="Number of previous defaults", ge=0, le=10)
    open_credit_lines: Optional[int] = Field(None, description="Number of open credit lines", ge=0, le=50)
    total_credit_lines: Optional[int] = Field(None, description="Total number of credit lines", ge=0, le=100)
    revolving_credit_balance: Optional[float] = Field(None, description="Revolving credit balance", ge=0)
    revolving_credit_limit: Optional[float] = Field(None, description="Revolving credit limit", ge=0)
    
    @validator('income', 'annual_income')
    def validate_income_realistic(cls, v):
        """Validate income is realistic."""
        if v is not None and (v < 1000 or v > 10000000):
            raise ValueError('Income must be between $1,000 and $10,000,000')
        return v
    
    @validator('debt_to_income')
    def validate_dti_realistic(cls, v):
        """Validate debt-to-income ratio is realistic."""
        if v > 0.8:
            raise ValueError('Debt-to-income ratio cannot exceed 80%')
        return v
    
    @validator('interest_rate')
    def validate_interest_rate_realistic(cls, v):
        """Validate interest rate is realistic."""
        if v > 0.40:  # 40% APR
            raise ValueError('Interest rate cannot exceed 40%')
        return v


class PredictionResult(BaseModel):
    """Individual prediction result."""
    probability_of_default: float = Field(..., description="Probability of default (0-1)", ge=0, le=1)
    loss_given_default: float = Field(..., description="Loss given default (0-1)", ge=0, le=1)
    expected_loss: float = Field(..., description="Expected loss (0-1)", ge=0, le=1)
    risk_rating: int = Field(..., description="Risk rating (1-10)", ge=1, le=10)
    risk_category: str = Field(..., description="Risk category description")
    decision: Dict[str, Any] = Field(..., description="Lending decision and reason")


class ConfidenceScores(BaseModel):
    """Model confidence scores."""
    pd_confidence: float = Field(..., description="PD model confidence (0-1)", ge=0, le=1)
    lgd_confidence: float = Field(..., description="LGD model confidence (0-1)", ge=0, le=1)


class EconomicContext(BaseModel):
    """Current economic context."""
    fed_rate: Optional[float] = Field(None, description="Federal funds rate")
    unemployment_rate: Optional[float] = Field(None, description="Unemployment rate")
    economic_stress: Optional[float] = Field(None, description="Economic stress index")


class CreditRiskResponse(BaseModel):
    """
    Credit risk prediction response schema.
    Returns comprehensive risk assessment results.
    """
    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "request_id": "req_abc123",
                "success": True,
                "predictions": {
                    "probability_of_default": 0.08,
                    "loss_given_default": 0.45,
                    "expected_loss": 0.036,
                    "risk_rating": 4,
                    "risk_category": "Medium Risk",
                    "decision": {
                        "decision": "APPROVE_WITH_CONDITIONS",
                        "reason": "Moderate risk - additional conditions apply",
                        "requires_review": False
                    }
                },
                "confidence_scores": {
                    "pd_confidence": 0.85,
                    "lgd_confidence": 0.78
                },
                "economic_context": {
                    "fed_rate": 5.25,
                    "unemployment_rate": 3.8,
                    "economic_stress": 0.3
                },
                "model_version": "v1.2.0",
                "response_time_ms": 45.3,
                "timestamp": "2024-01-15T10:30:45.123Z"
            }
        }
    )

    request_id: str = Field(..., description="Unique request identifier")
    success: bool = Field(..., description="Whether prediction was successful")
    predictions: PredictionResult = Field(..., description="Risk prediction results")
    confidence_scores: ConfidenceScores = Field(..., description="Model confidence scores")
    economic_context: EconomicContext = Field(..., description="Economic context data")
    model_version: Optional[str] = Field(None, description="Version of model used")
    response_time_ms: float = Field(..., description="Response time in milliseconds")
    timestamp: str = Field(..., description="Prediction timestamp (ISO format)")


class BatchPredictionRequest(BaseModel):
    """
    Batch prediction request schema.
    Validates multiple loan applications for batch processing.
    """
    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "applications": [
                    {
                        "customer_id": "BATCH_001",
                        "loan_amount": 15000.0,
                        "loan_term": 36,
                        "interest_rate": 0.11,
                        "income": 50000.0,
                        "debt_to_income": 0.30,
                        "credit_score": 700,
                        "employment_length": 3,
                        "home_ownership": "RENT",
                        "loan_purpose": "debt_consolidation"
                    }
                ],
                "batch_size": 50,
                "priority": "normal"
            }
        }
    )

    applications: List[CreditRiskRequest] = Field(
        ..., 
        description="List of credit applications to process",
        min_items=1,
        max_items=1000
    )
    batch_size: Optional[int] = Field(
        50, 
        description="Processing batch size",
        ge=1,
        le=100
    )
    priority: Optional[str] = Field("normal", description="Processing priority")
    
    @validator('applications')
    def validate_batch_size(cls, v):
        """Validate batch size is reasonable."""
        if len(v) > 1000:
            raise ValueError('Batch size cannot exceed 1000 applications')
        return v


class BatchPredictionResult(BaseModel):
    """Individual result in batch prediction."""
    application_index: int = Field(..., description="Index of application in batch")
    predictions: PredictionResult = Field(..., description="Prediction results")
    confidence_scores: ConfidenceScores = Field(..., description="Confidence scores")
    response_time_ms: float = Field(..., description="Individual response time")


class BatchPredictionError(BaseModel):
    """Error information for failed batch predictions."""
    application_index: int = Field(..., description="Index of failed application")
    error: str = Field(..., description="Error type")
    error_details: str = Field(..., description="Detailed error message")


class BatchPredictionResponse(BaseModel):
    """
    Batch prediction response schema.
    Returns results for multiple credit risk assessments.
    """
    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "request_id": "batch_xyz789",
                "success": True,
                "total_applications": 100,
                "successful_predictions": 98,
                "failed_predictions": 2,
                "results": [
                    {
                        "application_index": 0,
                        "predictions": {
                            "probability_of_default": 0.05,
                            "loss_given_default": 0.40,
                            "expected_loss": 0.02,
                            "risk_rating": 3,
                            "risk_category": "Low-Medium Risk",
                            "decision": {
                                "decision": "APPROVE",
                                "reason": "Low risk profile"
                            }
                        },
                        "confidence_scores": {"pd_confidence": 0.92, "lgd_confidence": 0.85},
                        "response_time_ms": 52.1
                    }
                ],
                "errors": [
                    {
                        "application_index": 5,
                        "error": "InvalidInput",
                        "error_details": "Credit score out of range"
                    }
                ],
                "response_time_ms": 5234.5,
                "timestamp": "2024-01-15T12:00:00Z"
            }
        }
    )

    request_id: str = Field(..., description="Unique batch request identifier")
    success: bool = Field(..., description="Whether batch processing was successful")
    total_applications: int = Field(..., description="Total number of applications processed")
    successful_predictions: int = Field(..., description="Number of successful predictions")
    failed_predictions: int = Field(..., description="Number of failed predictions")
    results: List[BatchPredictionResult] = Field(..., description="Successful prediction results")
    errors: List[BatchPredictionError] = Field(..., description="Failed prediction errors")
    response_time_ms: float = Field(..., description="Total batch response time")
    timestamp: str = Field(..., description="Batch completion timestamp")


class ErrorResponse(BaseModel):
    """Standard error response schema."""
    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "error": "InvalidInput",
                "message": "Input validation failed",
                "details": {
                    "credit_score": "Value must be between 300 and 850"
                },
                "request_id": "req_def456",
                "timestamp": "2024-01-15T10:35:10Z"
            }
        }
    )

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    request_id: Optional[str] = Field(None, description="Request identifier if available")
    timestamp: str = Field(..., description="Error timestamp")