"""
Database connection and utilities for RiskFlow Credit Risk MLOps Pipeline.
"""

import sqlite3
from contextlib import contextmanager
from typing import Generator, Optional, Dict, Any, List
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime, Boolean, JSON, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from datetime import datetime
import pandas as pd

from config.settings import get_settings
from config.logging_config import get_logger
from utils.helpers import get_utc_now

logger = get_logger(__name__)
settings = get_settings()

# SQLAlchemy setup
engine = create_engine(
    settings.database_url,
    echo=settings.database_echo,
    poolclass=StaticPool,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
    pool_pre_ping=True,
    pool_recycle=300
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# Database Models
class BaseModel(Base):
    __abstract__ = True
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    created_at = Column(DateTime, default=get_utc_now)
    updated_at = Column(DateTime, default=get_utc_now, onupdate=get_utc_now)


class CreditData(BaseModel):
    """Credit risk data table."""
    __tablename__ = "credit_data"
    
    customer_id = Column(String, index=True)
    loan_amount = Column(Float)
    loan_term = Column(Integer)
    interest_rate = Column(Float)
    income = Column(Float)
    debt_to_income = Column(Float)
    credit_score = Column(Integer)
    employment_length = Column(Integer)
    home_ownership = Column(String)
    loan_purpose = Column(String)
    default_probability = Column(Float)
    loss_given_default = Column(Float)
    is_default = Column(Boolean)


class ModelRegistry(BaseModel):
    """Model registry table for tracking model versions."""
    __tablename__ = "model_registry"
    
    model_name = Column(String, nullable=False, index=True)
    model_version = Column(String, nullable=False, unique=True, index=True)
    model_type = Column(String)  # PD, LGD, etc.
    model_path = Column(String, nullable=False)
    mlflow_run_id = Column(String)
    performance_metrics = Column(JSON)
    status = Column(String)  # active, archived, deprecated
    stage = Column(String, default='development', index=True)  # e.g., development, staging, production
    created_at = Column(DateTime, default=get_utc_now)
    updated_at = Column(DateTime, default=get_utc_now, onupdate=get_utc_now)


class PredictionLog(Base):
    """Prediction request logging table."""
    __tablename__ = "prediction_logs"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    request_id = Column(String, unique=True, index=True)
    model_name = Column(String)
    model_version = Column(String, index=True)
    input_features = Column(JSON)
    prediction_result = Column(JSON)
    confidence_score = Column(Float)
    response_time_ms = Column(Float)
    timestamp = Column(DateTime, default=get_utc_now)


class ModelPerformance(Base):
    """Model performance monitoring table."""
    __tablename__ = "model_performance"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    model_name = Column(String)
    model_version = Column(String)
    metric_name = Column(String)
    metric_value = Column(Float)
    data_window_start = Column(DateTime)
    data_window_end = Column(DateTime)
    timestamp = Column(DateTime, default=get_utc_now)


class IngestionLog(Base):
    __tablename__ = 'ingestion_logs'
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=get_utc_now)
    source = Column(String)
    status = Column(String)


def create_tables():
    """Create all database tables."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {str(e)}")
        raise


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Get database session with automatic cleanup.
    
    Yields:
        SQLAlchemy session instance
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {str(e)}")
        raise
    finally:
        session.close()


def get_db():
    """FastAPI dependency for database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class DatabaseManager:
    """Database operations manager."""
    
    def __init__(self):
        self.engine = engine
        self.SessionLocal = SessionLocal
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Get database session with automatic cleanup.
        
        Yields:
            SQLAlchemy session instance
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {str(e)}")
            raise
        finally:
            session.close()
    
    def health_check(self) -> bool:
        """
        Check database connectivity.
        
        Returns:
            True if database is accessible, False otherwise
        """
        try:
            with get_db_session() as session:
                session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            return False
    
    def insert_credit_data(self, data: List[Dict[str, Any]]) -> int:
        """
        Insert credit data records.
        
        Args:
            data: List of credit data dictionaries
        
        Returns:
            Number of records inserted
        """
        try:
            with get_db_session() as session:
                records = [CreditData(**record) for record in data]
                session.add_all(records)
                session.commit()
                logger.info(f"Inserted {len(records)} credit data records")
                return len(records)
        except Exception as e:
            logger.error(f"Failed to insert credit data: {str(e)}")
            raise
    
    def get_credit_data(
        self, 
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve credit data records.
        
        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
            filters: Dictionary of filters to apply
        
        Returns:
            List of credit data dictionaries
        """
        try:
            with get_db_session() as session:
                query = session.query(CreditData)
                
                # Apply filters if provided
                if filters:
                    for field, value in filters.items():
                        if hasattr(CreditData, field):
                            query = query.filter(getattr(CreditData, field) == value)
                
                # Apply pagination
                if offset:
                    query = query.offset(offset)
                if limit:
                    query = query.limit(limit)
                
                results = query.all()
                return [self._credit_data_to_dict(record) for record in results]
        except Exception as e:
            logger.error(f"Failed to retrieve credit data: {str(e)}")
            raise
    
    def get_credit_data_count(self) -> int:
        """
        Get total count of credit data records.
        
        Returns:
            Total number of credit records
        """
        try:
            with get_db_session() as session:
                count = session.query(CreditData).count()
                return count
        except Exception as e:
            logger.error(f"Failed to count credit data: {str(e)}")
            return 0
    
    def register_model(
        self,
        model_name: str,
        model_version: str,
        model_type: str,
        model_path: str,
        mlflow_run_id: str,
        performance_metrics: Dict[str, float]
    ) -> int:
        """
        Register a new model in the registry.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            model_type: Type of model (PD, LGD, etc.)
            model_path: Path to model artifacts
            mlflow_run_id: MLflow run ID
            performance_metrics: Model performance metrics
        
        Returns:
            Model registry ID
        """
        try:
            import json
            with get_db_session() as session:
                model_record = ModelRegistry(
                    model_name=model_name,
                    model_version=model_version,
                    model_type=model_type,
                    model_path=model_path,
                    mlflow_run_id=mlflow_run_id,
                    performance_metrics=json.dumps(performance_metrics),
                    status="active"
                )
                session.add(model_record)
                session.commit()
                logger.info(f"Registered model: {model_name} v{model_version}")
                return model_record.id
        except Exception as e:
            logger.error(f"Failed to register model: {str(e)}")
            raise
    
    def log_prediction(
        self,
        request_id: str,
        model_name: str,
        model_version: str,
        input_features: Dict[str, Any],
        prediction_result: Dict[str, Any],
        confidence_score: float,
        response_time_ms: float
    ) -> None:
        """
        Log a prediction request.
        
        Args:
            request_id: Unique request identifier
            model_name: Name of the model used
            model_version: Version of the model used
            input_features: Input features dictionary
            prediction_result: Prediction result dictionary
            confidence_score: Prediction confidence score
            response_time_ms: Response time in milliseconds
        """
        try:
            import json
            with get_db_session() as session:
                log_record = PredictionLog(
                    request_id=request_id,
                    model_name=model_name,
                    model_version=model_version,
                    input_features=json.dumps(input_features),
                    prediction_result=json.dumps(prediction_result),
                    confidence_score=confidence_score,
                    response_time_ms=response_time_ms
                )
                session.add(log_record)
                session.commit()
        except Exception as e:
            logger.error(f"Failed to log prediction: {str(e)}")
    
    def get_model_performance_history(
        self,
        model_name: str,
        days: int = 30
    ) -> pd.DataFrame:
        """
        Get model performance history.
        
        Args:
            model_name: Name of the model
            days: Number of days to look back
        
        Returns:
            DataFrame with performance history
        """
        try:
            from datetime import datetime, timedelta
            
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            with get_db_session() as session:
                results = session.query(ModelPerformance).filter(
                    ModelPerformance.model_name == model_name,
                    ModelPerformance.timestamp >= cutoff_date
                ).all()
                
                if not results:
                    return pd.DataFrame()
                
                data = []
                for record in results:
                    data.append({
                        'model_name': record.model_name,
                        'model_version': record.model_version,
                        'metric_name': record.metric_name,
                        'metric_value': record.metric_value,
                        'timestamp': record.timestamp
                    })
                
                return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Failed to get model performance history: {str(e)}")
            return pd.DataFrame()
    
    def log_api_usage(
        self,
        request_id: str,
        endpoint: str,
        success: bool,
        response_time_ms: float,
        error_message: Optional[str] = None,
        batch_size: Optional[int] = None
    ):
        """
        Log API usage statistics.
        
        Args:
            request_id: Unique request identifier
            endpoint: API endpoint path
            success: Whether request was successful
            response_time_ms: Response time in milliseconds
            error_message: Error message if applicable
            batch_size: Batch size for batch requests
        """
        try:
            # For now, just log to logger since we don't have API usage table
            log_data = {
                'request_id': request_id,
                'endpoint': endpoint,
                'success': success,
                'response_time_ms': response_time_ms,
                'error_message': error_message,
                'batch_size': batch_size,
                'timestamp': get_utc_now()
            }
            logger.info(f"API usage logged: {log_data}")
        except Exception as e:
            logger.error(f"Failed to log API usage: {str(e)}")
    
    def get_prediction_log(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get prediction log by request ID."""
        try:
            with get_db_session() as session:
                record = session.query(PredictionLog).filter(
                    PredictionLog.request_id == request_id
                ).first()
                
                if record:
                    return {
                        'request_id': record.request_id,
                        'model_name': record.model_name,
                        'model_version': record.model_version,
                        'timestamp': record.timestamp,
                        'response_time_ms': record.response_time_ms,
                        'success': True  # If logged, it was successful
                    }
                return None
        except Exception as e:
            logger.error(f"Failed to get prediction log: {str(e)}")
            return None
    
    def get_prediction_history(
        self,
        limit: int = 100,
        offset: int = 0,
        customer_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get prediction history with optional filtering."""
        try:
            with get_db_session() as session:
                query = session.query(PredictionLog)
                
                # Apply customer filter if provided (would need to parse input_features)
                # For now, just return recent predictions
                
                records = query.order_by(PredictionLog.timestamp.desc()).offset(offset).limit(limit).all()
                
                history = []
                for record in records:
                    history.append({
                        'request_id': record.request_id,
                        'model_name': record.model_name,
                        'model_version': record.model_version,
                        'timestamp': record.timestamp,
                        'response_time_ms': record.response_time_ms
                    })
                
                return history
        except Exception as e:
            logger.error(f"Failed to get prediction history: {str(e)}")
            return []
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            with get_db_session() as session:
                credit_count = session.query(CreditData).count()
                prediction_count = session.query(PredictionLog).count()
                
                # Get latest update timestamp
                latest_credit = session.query(CreditData).order_by(
                    CreditData.updated_at.desc()
                ).first()
                
                return {
                    'credit_data_count': credit_count,
                    'prediction_count': prediction_count,
                    'last_update': latest_credit.updated_at.isoformat() if latest_credit else None
                }
        except Exception as e:
            logger.error(f"Failed to get database stats: {str(e)}")
            return {'error': str(e)}
    
    def get_model_prediction_stats(self, model_version: str, days: int = 30) -> Dict[str, Any]:
        """Get prediction statistics for a model version."""
        try:
            from datetime import timedelta
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            with get_db_session() as session:
                records = session.query(PredictionLog).filter(
                    PredictionLog.model_version == model_version,
                    PredictionLog.timestamp >= cutoff_date
                ).all()
                
                if not records:
                    return {'total_predictions': 0}
                
                total_predictions = len(records)
                avg_response_time = sum(r.response_time_ms for r in records) / total_predictions
                
                return {
                    'total_predictions': total_predictions,
                    'avg_response_time_ms': avg_response_time,
                    'period_days': days
                }
        except Exception as e:
            logger.error(f"Failed to get model prediction stats: {str(e)}")
            return {'error': str(e)}
    
    def get_model_drift_metrics(
        self, 
        days: int = 30, 
        model_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get model drift metrics (placeholder implementation)."""
        try:
            # Placeholder implementation - would analyze prediction patterns
            return {
                'overall_drift_score': 0.05,
                'feature_drift': {'credit_score': 0.02, 'income': 0.08},
                'prediction_drift': {'pd_distribution': 0.03},
                'analysis_period_days': days
            }
        except Exception as e:
            logger.error(f"Failed to get drift metrics: {str(e)}")
            return {'error': str(e)}
    
    def _credit_data_to_dict(self, record: CreditData) -> Dict[str, Any]:
        """Convert CreditData object to dictionary."""
        return {
            'id': record.id,
            'customer_id': record.customer_id,
            'loan_amount': record.loan_amount,
            'loan_term': record.loan_term,
            'interest_rate': record.interest_rate,
            'income': record.income,
            'debt_to_income': record.debt_to_income,
            'credit_score': record.credit_score,
            'employment_length': record.employment_length,
            'home_ownership': record.home_ownership,
            'loan_purpose': record.loan_purpose,
            'default_probability': record.default_probability,
            'loss_given_default': record.loss_given_default,
            'is_default': record.is_default,
            'created_at': record.created_at,
            'updated_at': record.updated_at
        }


# Global database manager instance
db_manager = DatabaseManager()


def initialize_database():
    """Initialize database with tables and basic setup."""
    try:
        create_tables()
        logger.info("Database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        return False