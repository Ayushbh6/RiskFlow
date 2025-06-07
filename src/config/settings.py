"""
Application configuration management for RiskFlow Credit Risk MLOps Pipeline.
"""

import os
from typing import Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Application configuration settings with environment variable support."""
    
    # Application
    app_name: str = "RiskFlow"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: str = "development"
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    api_reload: bool = True
    
    # Database Configuration
    database_url: str = "sqlite:///./data/riskflow.db"
    database_echo: bool = False
    database_pool_size: int = 5
    database_max_overflow: int = 10
    
    # MLflow Configuration
    mlflow_tracking_uri: str = "sqlite:///./mlflow.db"
    mlflow_experiment_name: str = "credit_risk_models"
    mlflow_artifact_location: str = "./mlflow/artifacts"
    
    @property
    def _base_path(self):
        """Get base project path."""
        return Path(__file__).parent.parent.parent
    
    def __init__(self, **kwargs):
        """Initialize settings with absolute paths."""
        super().__init__(**kwargs)
        # Update paths to be absolute
        base_path = self._base_path
        self.database_url = f"sqlite:///{base_path}/data/riskflow.db"
        self.mlflow_tracking_uri = f"sqlite:///{base_path}/mlflow.db"
        self.mlflow_artifact_location = f"{base_path}/mlflow/artifacts"
    
    # LLM Configuration
    llm_provider: str = "openai"  # "openai" or "ollama"
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-3.5-turbo"
    openai_max_tokens: int = 2000
    openai_temperature: float = 0.1
    openai_timeout: float = 30.0
    
    # Ollama Configuration
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "gemma3:4b"
    ollama_timeout: float = 60.0
    
    # Redis Configuration (Optional)
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_enabled: bool = False
    
    # Model Configuration
    model_retrain_threshold: float = 0.05
    model_drift_threshold: float = 0.1
    model_performance_threshold: float = 0.8
    model_max_age_days: int = 30
    
    # Data Configuration
    data_dir: str = "./data"
    raw_data_dir: str = "./data/raw"
    processed_data_dir: str = "./data/processed"
    model_artifacts_dir: str = "./data/models"
    cache_dir: str = "./data/cache"
    
    # Logging Configuration
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: str = "./logs/riskflow.log"
    log_max_size: int = 10485760  # 10MB
    log_backup_count: int = 5
    
    # Dashboard Configuration
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8501
    dashboard_title: str = "RiskFlow - Credit Risk MLOps Dashboard"
    
    # Security Configuration
    api_key_header: str = "X-API-Key"
    api_rate_limit: str = "100/minute"
    cors_origins: list = ["http://localhost:3000", "http://localhost:8501"]
    
    # Additional environment variables
    fred_api_key: Optional[str] = None
    tavily_api_key: Optional[str] = None
    redis_url: Optional[str] = None
    model_retrain_interval_hours: int = 24
    cache_ttl_seconds: int = 3600
    cache_max_size: int = 1000
    rate_limit_per_minute: int = 100
    max_batch_size: int = 100
    
    @field_validator("openai_api_key")
    @classmethod
    def validate_openai_key(cls, v):
        if v is None:
            # Try to get from environment
            v = os.getenv("OPENAI_API_KEY")
        if v is None:
            print("Warning: OpenAI API key not provided. LLM features will be disabled.")
        return v
    
    @field_validator("data_dir", "raw_data_dir", "processed_data_dir", "model_artifacts_dir", "cache_dir")
    @classmethod
    def create_directories(cls, v):
        Path(v).mkdir(parents=True, exist_ok=True)
        return v
    
    @field_validator("log_file")
    @classmethod
    def create_log_directory(cls, v):
        Path(v).parent.mkdir(parents=True, exist_ok=True)
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings instance."""
    return settings


def get_database_url() -> str:
    """Get database connection URL."""
    return settings.database_url


def get_mlflow_config() -> dict:
    """Get MLflow configuration dictionary."""
    return {
        "tracking_uri": settings.mlflow_tracking_uri,
        "experiment_name": settings.mlflow_experiment_name,
        "artifact_location": settings.mlflow_artifact_location,
    }


def get_openai_config() -> dict:
    """Get OpenAI configuration dictionary."""
    return {
        "api_key": settings.openai_api_key,
        "model": settings.openai_model,
        "max_tokens": settings.openai_max_tokens,
        "temperature": settings.openai_temperature,
        "timeout": settings.openai_timeout,
    }


def get_ollama_config() -> dict:
    """Get Ollama configuration dictionary."""
    return {
        "host": settings.ollama_host,
        "model": settings.ollama_model,
        "timeout": settings.ollama_timeout,
    }


def get_llm_config() -> dict:
    """Get LLM configuration based on provider."""
    if settings.llm_provider.lower() == "ollama":
        return {"provider": "ollama", **get_ollama_config()}
    else:
        return {"provider": "openai", **get_openai_config()}


def is_production() -> bool:
    """Check if running in production environment."""
    return settings.environment.lower() == "production"


def is_development() -> bool:
    """Check if running in development environment."""
    return settings.environment.lower() == "development"


def is_redis_enabled() -> bool:
    """Check if Redis caching is enabled."""
    return settings.redis_enabled


def get_feature_config() -> dict:
    """Get feature engineering configuration."""
    return {
        "outlier_threshold": 3.0,
        "missing_value_strategy": "median",
        "categorical_encoding": "target",
        "scaling_method": "robust",
        "feature_selection_threshold": 0.01,
        "max_features": 50,
    }


# Environment-specific configurations
DEVELOPMENT_CONFIG = {
    "debug": True,
    "api_reload": True,
    "log_level": "DEBUG",
    "database_echo": True,
}

PRODUCTION_CONFIG = {
    "debug": False,
    "api_reload": False,
    "log_level": "INFO",
    "database_echo": False,
    "api_workers": 4,
}

TESTING_CONFIG = {
    "debug": True,
    "database_url": "sqlite:///:memory:",
    "log_level": "DEBUG",
    "mlflow_tracking_uri": "./test_mlflow",
}