"""
LLM-powered credit risk analyzer with OpenAI and Ollama support.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import asyncio

from .openai_client import get_openai_client, is_openai_available, OpenAIClient
from .ollama_client import get_ollama_client, is_ollama_available, OllamaClient
from .prompt_templates import PromptTemplates, customize_prompt_for_audience, add_context_to_prompt
from ..config.settings import get_llm_config, settings
from ..utils.exceptions import LLMError, ConfigurationException

logger = logging.getLogger(__name__)


class RiskAnalyzer:
    """
    LLM-powered credit risk analyzer with multi-provider support.
    
    Supports both OpenAI and Ollama as LLM providers with automatic fallback.
    Provides comprehensive credit risk analysis, model documentation, and insights.
    """
    
    def __init__(self):
        """Initialize risk analyzer with configured LLM provider."""
        self.config = get_llm_config()
        self.provider = self.config.get("provider", "openai").lower()
        
        # Initialize clients
        self.openai_client: Optional[OpenAIClient] = None
        self.ollama_client: Optional[OllamaClient] = None
        
        # Set up primary and fallback providers
        self._setup_providers()
        
        logger.info(f"RiskAnalyzer initialized with provider: {self.provider}")
    
    def _setup_providers(self) -> None:
        """Set up LLM providers with fallback logic."""
        if self.provider == "openai":
            # Primary: OpenAI, Fallback: Ollama
            if is_openai_available():
                try:
                    self.openai_client = get_openai_client()
                    logger.info("OpenAI client initialized as primary provider")
                except Exception as e:
                    logger.warning(f"Failed to initialize OpenAI client: {e}")
            
            # Set up Ollama as fallback
            if is_ollama_available():
                try:
                    self.ollama_client = get_ollama_client()
                    logger.info("Ollama client initialized as fallback provider")
                except Exception as e:
                    logger.warning(f"Failed to initialize Ollama client: {e}")
        
        else:
            # Primary: Ollama, Fallback: OpenAI
            if is_ollama_available():
                try:
                    self.ollama_client = get_ollama_client()
                    logger.info("Ollama client initialized as primary provider")
                except Exception as e:
                    logger.warning(f"Failed to initialize Ollama client: {e}")
            
            # Set up OpenAI as fallback
            if is_openai_available():
                try:
                    self.openai_client = get_openai_client()
                    logger.info("OpenAI client initialized as fallback provider")
                except Exception as e:
                    logger.warning(f"Failed to initialize OpenAI client: {e}")
        
        # Validate at least one provider is available
        if not self.openai_client and not self.ollama_client:
            raise ConfigurationException("No LLM providers available. Please configure OpenAI or Ollama.")
    
    def _get_primary_client(self) -> Union[OpenAIClient, OllamaClient]:
        """Get the primary LLM client."""
        if self.provider == "openai" and self.openai_client:
            return self.openai_client
        elif self.provider == "ollama" and self.ollama_client:
            return self.ollama_client
        else:
            raise LLMError(f"Primary provider {self.provider} not available")
    
    def _get_fallback_client(self) -> Optional[Union[OpenAIClient, OllamaClient]]:
        """Get the fallback LLM client."""
        if self.provider == "openai" and self.ollama_client:
            return self.ollama_client
        elif self.provider == "ollama" and self.openai_client:
            return self.openai_client
        else:
            return None
    
    def _make_llm_request(
        self,
        messages: List[Dict[str, str]],
        use_fallback: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make LLM request with automatic fallback.
        
        Args:
            messages: Chat messages for LLM
            use_fallback: Whether to use fallback provider on failure
            **kwargs: Additional parameters for LLM call
        
        Returns:
            LLM response dictionary
        
        Raises:
            LLMError: If all providers fail
        """
        # Try primary provider
        try:
            primary_client = self._get_primary_client()
            response = primary_client.chat_completion(messages, **kwargs)
            response["provider_used"] = self.provider
            return response
        except Exception as e:
            logger.warning(f"Primary provider {self.provider} failed: {e}")
            
            if not use_fallback:
                raise LLMError(f"Primary provider failed: {e}")
        
        # Try fallback provider
        fallback_client = self._get_fallback_client()
        if fallback_client:
            try:
                fallback_provider = "ollama" if self.provider == "openai" else "openai"
                logger.info(f"Attempting fallback to {fallback_provider}")
                
                response = fallback_client.chat_completion(messages, **kwargs)
                response["provider_used"] = fallback_provider
                return response
            except Exception as e:
                logger.error(f"Fallback provider also failed: {e}")
                raise LLMError(f"All LLM providers failed. Primary: {self.provider}, Fallback: {fallback_provider}")
        else:
            raise LLMError(f"No fallback provider available")
    
    async def _make_async_llm_request(
        self,
        messages: List[Dict[str, str]],
        use_fallback: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make async LLM request with automatic fallback.
        
        Args:
            messages: Chat messages for LLM
            use_fallback: Whether to use fallback provider on failure
            **kwargs: Additional parameters for LLM call
        
        Returns:
            LLM response dictionary
        
        Raises:
            LLMError: If all providers fail
        """
        # Try primary provider
        try:
            primary_client = self._get_primary_client()
            response = await primary_client.async_chat_completion(messages, **kwargs)
            response["provider_used"] = self.provider
            return response
        except Exception as e:
            logger.warning(f"Primary provider {self.provider} failed: {e}")
            
            if not use_fallback:
                raise LLMError(f"Primary provider failed: {e}")
        
        # Try fallback provider
        fallback_client = self._get_fallback_client()
        if fallback_client:
            try:
                fallback_provider = "ollama" if self.provider == "openai" else "openai"
                logger.info(f"Attempting async fallback to {fallback_provider}")
                
                response = await fallback_client.async_chat_completion(messages, **kwargs)
                response["provider_used"] = fallback_provider
                return response
            except Exception as e:
                logger.error(f"Async fallback provider also failed: {e}")
                raise LLMError(f"All async LLM providers failed")
        else:
            raise LLMError(f"No async fallback provider available")
    
    def analyze_credit_risk(
        self,
        borrower_data: Dict[str, Any],
        model_prediction: Dict[str, Any],
        market_context: Optional[Dict[str, Any]] = None,
        audience: str = "business",
        additional_context: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate comprehensive credit risk analysis.
        
        Args:
            borrower_data: Borrower information and financial metrics
            model_prediction: Model prediction results (PD, LGD, etc.)
            market_context: Current market conditions and economic indicators
            audience: Target audience for the analysis
            additional_context: Additional context to include
            **kwargs: Additional parameters for LLM call
        
        Returns:
            Dictionary containing risk analysis and metadata
        """
        try:
            # Generate prompt
            messages = PromptTemplates.risk_analysis_prompt(
                borrower_data=borrower_data,
                model_prediction=model_prediction,
                market_context=market_context
            )
            
            # Customize for audience
            messages = customize_prompt_for_audience(messages, audience)
            
            # Add additional context if provided
            if additional_context:
                messages = add_context_to_prompt(messages, additional_context)
            
            # Make LLM request
            response = self._make_llm_request(messages, **kwargs)
            
            # Structure the response
            result = {
                "analysis": response["content"],
                "borrower_id": borrower_data.get("borrower_id"),
                "analysis_type": "credit_risk",
                "audience": audience,
                "model_prediction": model_prediction,
                "provider_used": response.get("provider_used"),
                "usage": response.get("usage"),
                "created_at": response.get("created_at"),
                "metadata": {
                    "model": response.get("model"),
                    "finish_reason": response.get("finish_reason"),
                    "confidence": model_prediction.get("confidence"),
                }
            }
            
            logger.info(f"Credit risk analysis completed for borrower {borrower_data.get('borrower_id')}")
            return result
            
        except Exception as e:
            logger.error(f"Credit risk analysis failed: {e}")
            raise LLMError(f"Failed to generate credit risk analysis: {str(e)}")
    
    def generate_model_documentation(
        self,
        model_info: Dict[str, Any],
        performance_metrics: Dict[str, Any],
        feature_importance: List[Dict[str, Any]],
        audience: str = "technical",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate comprehensive model documentation.
        
        Args:
            model_info: Model metadata and configuration
            performance_metrics: Model performance statistics
            feature_importance: Feature importance rankings
            audience: Target audience for documentation
            **kwargs: Additional parameters for LLM call
        
        Returns:
            Dictionary containing model documentation and metadata
        """
        try:
            # Generate prompt
            messages = PromptTemplates.model_documentation_prompt(
                model_info=model_info,
                performance_metrics=performance_metrics,
                feature_importance=feature_importance
            )
            
            # Customize for audience
            messages = customize_prompt_for_audience(messages, audience)
            
            # Make LLM request
            response = self._make_llm_request(messages, **kwargs)
            
            # Structure the response
            result = {
                "documentation": response["content"],
                "model_id": model_info.get("model_id"),
                "model_version": model_info.get("version"),
                "documentation_type": "model_documentation",
                "audience": audience,
                "provider_used": response.get("provider_used"),
                "usage": response.get("usage"),
                "created_at": response.get("created_at"),
                "metadata": {
                    "model": response.get("model"),
                    "finish_reason": response.get("finish_reason"),
                    "performance_metrics": performance_metrics,
                }
            }
            
            logger.info(f"Model documentation generated for model {model_info.get('model_id')}")
            return result
            
        except Exception as e:
            logger.error(f"Model documentation generation failed: {e}")
            raise LLMError(f"Failed to generate model documentation: {str(e)}")
    
    def explain_prediction(
        self,
        prediction_details: Dict[str, Any],
        feature_contributions: List[Dict[str, Any]],
        borrower_profile: Dict[str, Any],
        audience: str = "business",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate explanation for model prediction.
        
        Args:
            prediction_details: Detailed prediction results
            feature_contributions: SHAP values or feature contributions
            borrower_profile: Borrower characteristics
            audience: Target audience for explanation
            **kwargs: Additional parameters for LLM call
        
        Returns:
            Dictionary containing prediction explanation and metadata
        """
        try:
            # Generate prompt
            messages = PromptTemplates.model_explanation_prompt(
                prediction_details=prediction_details,
                feature_contributions=feature_contributions,
                borrower_profile=borrower_profile
            )
            
            # Customize for audience
            messages = customize_prompt_for_audience(messages, audience)
            
            # Make LLM request
            response = self._make_llm_request(messages, **kwargs)
            
            # Structure the response
            result = {
                "explanation": response["content"],
                "borrower_id": borrower_profile.get("borrower_id"),
                "prediction_id": prediction_details.get("prediction_id"),
                "explanation_type": "prediction_explanation",
                "audience": audience,
                "provider_used": response.get("provider_used"),
                "usage": response.get("usage"),
                "created_at": response.get("created_at"),
                "metadata": {
                    "model": response.get("model"),
                    "finish_reason": response.get("finish_reason"),
                    "prediction_details": prediction_details,
                }
            }
            
            logger.info(f"Prediction explanation generated for borrower {borrower_profile.get('borrower_id')}")
            return result
            
        except Exception as e:
            logger.error(f"Prediction explanation generation failed: {e}")
            raise LLMError(f"Failed to generate prediction explanation: {str(e)}")
    
    async def analyze_credit_risk_async(
        self,
        borrower_data: Dict[str, Any],
        model_prediction: Dict[str, Any],
        market_context: Optional[Dict[str, Any]] = None,
        audience: str = "business",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Asynchronously generate comprehensive credit risk analysis.
        
        Args:
            borrower_data: Borrower information and financial metrics
            model_prediction: Model prediction results (PD, LGD, etc.)
            market_context: Current market conditions and economic indicators
            audience: Target audience for the analysis
            **kwargs: Additional parameters for LLM call
        
        Returns:
            Dictionary containing risk analysis and metadata
        """
        try:
            # Generate prompt
            messages = PromptTemplates.risk_analysis_prompt(
                borrower_data=borrower_data,
                model_prediction=model_prediction,
                market_context=market_context
            )
            
            # Customize for audience
            messages = customize_prompt_for_audience(messages, audience)
            
            # Make async LLM request
            response = await self._make_async_llm_request(messages, **kwargs)
            
            # Structure the response
            result = {
                "analysis": response["content"],
                "borrower_id": borrower_data.get("borrower_id"),
                "analysis_type": "credit_risk_async",
                "audience": audience,
                "model_prediction": model_prediction,
                "provider_used": response.get("provider_used"),
                "usage": response.get("usage"),
                "created_at": response.get("created_at"),
                "metadata": {
                    "model": response.get("model"),
                    "finish_reason": response.get("finish_reason"),
                    "confidence": model_prediction.get("confidence"),
                }
            }
            
            logger.info(f"Async credit risk analysis completed for borrower {borrower_data.get('borrower_id')}")
            return result
            
        except Exception as e:
            logger.error(f"Async credit risk analysis failed: {e}")
            raise LLMError(f"Failed to generate async credit risk analysis: {str(e)}")
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test LLM provider connections.
        
        Returns:
            Dictionary with connection test results
        """
        results = {
            "primary_provider": self.provider,
            "openai_available": False,
            "ollama_available": False,
            "primary_working": False,
            "fallback_working": False,
        }
        
        # Test OpenAI
        if self.openai_client:
            try:
                results["openai_available"] = self.openai_client.test_connection()
            except Exception as e:
                logger.warning(f"OpenAI connection test failed: {e}")
        
        # Test Ollama
        if self.ollama_client:
            try:
                results["ollama_available"] = self.ollama_client.test_connection()
            except Exception as e:
                logger.warning(f"Ollama connection test failed: {e}")
        
        # Test primary provider
        if self.provider == "openai":
            results["primary_working"] = results["openai_available"]
            results["fallback_working"] = results["ollama_available"]
        else:
            results["primary_working"] = results["ollama_available"]
            results["fallback_working"] = results["openai_available"]
        
        return results
    
    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get information about configured LLM providers.
        
        Returns:
            Dictionary with provider information
        """
        info = {
            "primary_provider": self.provider,
            "providers": {}
        }
        
        if self.openai_client:
            info["providers"]["openai"] = self.openai_client.get_model_info()
        
        if self.ollama_client:
            info["providers"]["ollama"] = self.ollama_client.get_model_info()
        
        return info


# Global analyzer instance
_risk_analyzer: Optional[RiskAnalyzer] = None


def get_risk_analyzer() -> RiskAnalyzer:
    """Get or create risk analyzer instance."""
    global _risk_analyzer
    if _risk_analyzer is None:
        _risk_analyzer = RiskAnalyzer()
    return _risk_analyzer


def is_llm_available() -> bool:
    """Check if any LLM provider is available."""
    return is_openai_available() or is_ollama_available() 