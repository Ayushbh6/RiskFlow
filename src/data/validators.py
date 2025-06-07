"""
Data validation for RiskFlow Credit Risk MLOps Pipeline.
Ensures all data is real, accurate, and meets quality standards.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pydantic import BaseModel, field_validator, ValidationError, ConfigDict
import json

from config.settings import get_settings
from config.logging_config import get_logger
from data.data_sources import DataProvider
from utils.exceptions import DataValidationError
from utils.helpers import get_utc_now

logger = get_logger(__name__)
settings = get_settings()


class EconomicFactorsSchema(BaseModel):
    """Schema for validating real economic factors data."""
    
    model_config = ConfigDict(extra="forbid")

    fed_funds_rate: Optional[float] = None
    unemployment_rate: Optional[float] = None
    data_timestamp: str
    data_source: str
    
    @field_validator('fed_funds_rate')
    @classmethod
    def validate_fed_rate(cls, v):
        if v is not None:
            if not (0.0 <= v <= 20.0):  # Reasonable range for Fed funds rate
                raise ValueError(f"Fed funds rate {v} outside reasonable range [0, 20]")
        return v
    
    @field_validator('unemployment_rate')
    @classmethod
    def validate_unemployment(cls, v):
        if v is not None:
            if not (0.0 <= v <= 30.0):  # Reasonable range for unemployment
                raise ValueError(f"Unemployment rate {v} outside reasonable range [0, 30]")
        return v
    
    @field_validator('data_timestamp')
    @classmethod
    def validate_timestamp(cls, v):
        try:
            # Handle multiple timestamp formats
            if v.endswith('Z'):
                timestamp = datetime.fromisoformat(v.replace('Z', '+00:00'))
            elif '+' in v or v.endswith('00:00'):
                timestamp = datetime.fromisoformat(v)
            else:
                # Try parsing without timezone info
                timestamp = datetime.fromisoformat(v)
            
            # Data shouldn't be older than 7 days for real-time system
            max_age = get_utc_now() - timedelta(days=7)
            if timestamp < max_age:
                raise ValueError(f"Data timestamp {v} is too old (>{7} days)")
            return v
        except Exception as e:
            raise ValueError(f"Invalid timestamp format: {v} - {str(e)}")
    
    @field_validator('data_source')
    @classmethod
    def validate_source(cls, v):
        valid_sources = ['real_market_data', 'fred_api', 'treasury_api', 'tavily_search']
        if v not in valid_sources:
            raise ValueError(f"Invalid data source: {v}. Must be one of {valid_sources}")
        return v


class CreditDataSchema(BaseModel):
    """Schema for validating credit risk data."""
    
    model_config = ConfigDict(extra="forbid")

    customer_id: str
    loan_amount: float
    loan_term: int
    interest_rate: float
    income: float
    debt_to_income: float
    credit_score: int
    employment_length: int
    home_ownership: str
    loan_purpose: str
    
    @field_validator('loan_amount')
    @classmethod
    def validate_loan_amount(cls, v):
        if not (1000 <= v <= 10000000):  # $1K to $10M range
            raise ValueError(f"Loan amount {v} outside reasonable range")
        return v
    
    @field_validator('loan_term')
    @classmethod
    def validate_loan_term(cls, v):
        if v not in [12, 24, 36, 48, 60, 84, 120, 180, 240, 360]:  # Common loan terms
            raise ValueError(f"Invalid loan term: {v} months")
        return v
    
    @field_validator('interest_rate')
    @classmethod
    def validate_interest_rate(cls, v):
        if not (0.01 <= v <= 0.50):  # 1% to 50% range
            raise ValueError(f"Interest rate {v} outside reasonable range")
        return v
    
    @field_validator('income')
    @classmethod
    def validate_income(cls, v):
        if not (10000 <= v <= 10000000):  # $10K to $10M range
            raise ValueError(f"Income {v} outside reasonable range")
        return v
    
    @field_validator('debt_to_income')
    @classmethod
    def validate_dti(cls, v):
        if not (0.0 <= v <= 2.0):  # 0% to 200% DTI
            raise ValueError(f"Debt-to-income ratio {v} outside reasonable range")
        return v
    
    @field_validator('credit_score')
    @classmethod
    def validate_credit_score(cls, v):
        if not (300 <= v <= 850):  # FICO score range
            raise ValueError(f"Credit score {v} outside valid FICO range [300, 850]")
        return v
    
    @field_validator('employment_length')
    @classmethod
    def validate_employment(cls, v):
        if not (0 <= v <= 50):  # 0 to 50 years
            raise ValueError(f"Employment length {v} outside reasonable range")
        return v
    
    @field_validator('home_ownership')
    @classmethod
    def validate_home_ownership(cls, v):
        valid_types = ['RENT', 'OWN', 'MORTGAGE', 'OTHER']
        if v.upper() not in valid_types:
            raise ValueError(f"Invalid home ownership: {v}")
        return v.upper()
    
    @field_validator('loan_purpose')
    @classmethod
    def validate_loan_purpose(cls, v):
        valid_purposes = [
            'DEBT_CONSOLIDATION', 'CREDIT_CARD', 'HOME_IMPROVEMENT', 
            'MAJOR_PURCHASE', 'MEDICAL', 'VACATION', 'WEDDING',
            'MOVING', 'HOUSE', 'CAR', 'OTHER'
        ]
        if v.upper() not in valid_purposes:
            raise ValueError(f"Invalid loan purpose: {v}")
        return v.upper()


class DataQualityValidator:
    """Comprehensive data quality validation for real financial data."""
    
    def __init__(self):
        self.validation_rules = {
            'completeness_threshold': 0.95,  # 95% completeness required
            'freshness_hours': 24,           # Data must be <24 hours old
            'outlier_std_threshold': 3.0,    # 3 standard deviations for outliers
        }
    
    def validate_economic_data(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate real economic data quality.
        
        Args:
            data: Economic data dictionary
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        try:
            # Validate using Pydantic schema
            EconomicFactorsSchema(**data)
            
            # Additional business logic validation
            if not self._check_data_freshness(data.get('data_timestamp')):
                errors.append("Economic data is too old for real-time system")
            
            if not self._check_data_completeness(data):
                errors.append("Economic data incomplete - missing critical indicators")
            
            # Validate data source authenticity
            if not self._validate_data_source(data.get('data_source')):
                errors.append("Economic data source not verified as real/authentic")
            
            logger.info(f"Economic data validation: {len(errors)} errors found")
            return len(errors) == 0, errors
            
        except ValidationError as e:
            errors.extend([str(error) for error in e.errors()])
            logger.error(f"Economic data validation failed: {errors}")
            return False, errors
        except Exception as e:
            error_msg = f"Economic data validation error: {str(e)}"
            logger.error(error_msg)
            return False, [error_msg]
    
    def validate_credit_data(self, data: List[Dict[str, Any]]) -> Tuple[bool, List[str], List[int]]:
        """
        Validate credit risk data quality.
        
        Args:
            data: List of credit data records
        
        Returns:
            Tuple of (is_valid, error_messages, invalid_record_indices)
        """
        errors = []
        invalid_indices = []
        
        if not data:
            return False, ["No credit data provided"], []
        
        try:
            for idx, record in enumerate(data):
                try:
                    CreditDataSchema(**record)
                except ValidationError as e:
                    invalid_indices.append(idx)
                    errors.extend([f"Record {idx}: {str(error)}" for error in e.errors()])
            
            # Check overall data quality
            valid_records = len(data) - len(invalid_indices)
            completeness_ratio = valid_records / len(data)
            
            if completeness_ratio < self.validation_rules['completeness_threshold']:
                errors.append(f"Data completeness {completeness_ratio:.2%} below threshold {self.validation_rules['completeness_threshold']:.2%}")
            
            # Check for suspicious patterns (potential fake data)
            if self._detect_suspicious_patterns(data):
                errors.append("Suspicious data patterns detected - possible non-authentic data")
            
            logger.info(f"Credit data validation: {len(errors)} errors, {len(invalid_indices)} invalid records")
            return len(errors) == 0, errors, invalid_indices
            
        except Exception as e:
            error_msg = f"Credit data validation error: {str(e)}"
            logger.error(error_msg)
            return False, [error_msg], list(range(len(data)))
    
    def validate_api_response(self, response_data: Any, expected_schema: str) -> Tuple[bool, List[str]]:
        """
        Validate API response data quality and authenticity.
        
        Args:
            response_data: API response data
            expected_schema: Expected schema type
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        try:
            if response_data is None:
                return False, ["API response is None - data provider unavailable"]
            
            if expected_schema == 'economic':
                return self.validate_economic_data(response_data)
            elif expected_schema == 'credit':
                if isinstance(response_data, list):
                    is_valid, errs, _ = self.validate_credit_data(response_data)
                    return is_valid, errs
                else:
                    return False, ["Credit data must be a list of records"]
            else:
                return False, [f"Unknown validation schema: {expected_schema}"]
                
        except Exception as e:
            error_msg = f"API response validation error: {str(e)}"
            logger.error(error_msg)
            return False, [error_msg]
    
    def _check_data_freshness(self, timestamp_str: Optional[str]) -> bool:
        """Check if data timestamp is fresh enough."""
        if not timestamp_str:
            return False
        
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            max_age = get_utc_now() - timedelta(hours=self.validation_rules['freshness_hours'])
            return timestamp >= max_age
        except Exception:
            return False
    
    def _check_data_completeness(self, data: Dict[str, Any]) -> bool:
        """Check if economic data has sufficient completeness."""
        required_fields = ['data_timestamp', 'data_source']
        optional_indicators = ['fed_funds_rate', 'unemployment_rate', 'avg_interest_rate']
        
        # Must have required fields
        for field in required_fields:
            if field not in data or data[field] is None:
                return False
        
        # Must have at least one economic indicator
        indicator_count = sum(1 for field in optional_indicators if field in data and data[field] is not None)
        return indicator_count >= 1
    
    def _validate_data_source(self, source: Optional[str]) -> bool:
        """Validate that data source is authentic/real."""
        if not source:
            return False
        
        authentic_sources = [
            'real_market_data', 'fred_api', 'treasury_api', 
            'tavily_search', 'bls_api', 'government_api'
        ]
        return source in authentic_sources
    
    def _detect_suspicious_patterns(self, data: List[Dict[str, Any]]) -> bool:
        """Detect patterns that might indicate fake/synthetic data."""
        if len(data) < 10:  # Need sufficient data for pattern detection
            return False
        
        try:
            df = pd.DataFrame(data)
            
            # Check for unrealistic uniformity in distributions
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                if col in df.columns:
                    values = df[col].dropna()
                    if len(values) > 5:
                        # Check for too-perfect normal distributions (sign of synthetic data)
                        std_dev = values.std()
                        if std_dev == 0:  # All values identical
                            return True
                        
                        # Check for suspiciously round numbers
                        round_numbers = values[values == values.round()].count()
                        if round_numbers / len(values) > 0.9:  # >90% round numbers suspicious
                            return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Could not perform suspicious pattern detection: {str(e)}")
            return False


def validate_real_time_data(data: Any, data_type: str) -> Dict[str, Any]:
    """
    Main validation function for real-time data.
    
    Args:
        data: Data to validate
        data_type: Type of data ('economic', 'credit', 'market')
    
    Returns:
        Validation result dictionary
    """
    validator = DataQualityValidator()
    
    try:
        if data_type == 'economic':
            is_valid, errors = validator.validate_economic_data(data)
        elif data_type == 'credit':
            is_valid, errors, invalid_indices = validator.validate_credit_data(data)
        else:
            is_valid, errors = False, [f"Unknown data type: {data_type}"]
        
        result = {
            'is_valid': is_valid,
            'errors': errors,
            'data_type': data_type,
            'validation_timestamp': get_utc_now().isoformat(),
            'validator_version': '1.0.0'
        }
        
        if data_type == 'credit' and 'invalid_indices' in locals():
            result['invalid_record_indices'] = invalid_indices
        
        if not is_valid:
            logger.warning(f"Data validation failed for {data_type}: {errors}")
        else:
            logger.info(f"Data validation successful for {data_type}")
        
        return result
        
    except Exception as e:
        error_msg = f"Data validation system error: {str(e)}"
        logger.error(error_msg)
        return {
            'is_valid': False,
            'errors': [error_msg],
            'data_type': data_type,
            'validation_timestamp': get_utc_now().isoformat(),
            'validator_version': '1.0.0'
        }