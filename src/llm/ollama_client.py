"""
Ollama API client for credit risk analysis and documentation generation.
"""

import json
import hashlib
import logging
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime
import asyncio

import ollama
from ollama import Client, AsyncClient, ResponseError

from ..config.settings import get_ollama_config, settings
from ..utils.cache import get_cache_client
from ..utils.exceptions import LLMError, ConfigurationException
from ..utils.helpers import get_utc_now

logger = logging.getLogger(__name__)


class OllamaClient:
    """Ollama API client with caching, error handling, and retry logic."""
    
    def __init__(self):
        """Initialize Ollama client with configuration."""
        self.config = get_ollama_config()
        self.cache_client = get_cache_client()
        self.cache_ttl = 3600  # 1 hour cache TTL
        
        # Initialize synchronous client
        self.client = Client(
            host=self.config["host"],
            timeout=self.config.get("timeout", 60.0),
        )
        
        # Initialize asynchronous client
        self.async_client = AsyncClient(
            host=self.config["host"],
            timeout=self.config.get("timeout", 60.0),
        )
        
        logger.info(f"Ollama client initialized with model: {self.config['model']}")
    
    def _generate_cache_key(self, messages: List[Dict], **kwargs) -> str:
        """Generate cache key for request."""
        cache_data = {
            "messages": messages,
            "model": self.config["model"],
            **kwargs
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return f"ollama:{hashlib.md5(cache_string.encode()).hexdigest()}"
    
    def _get_cached_response(self, cache_key: str) -> Optional[Dict]:
        """Get cached response if available."""
        if not self.cache_client:
            return None
        
        try:
            cached = self.cache_client.get(cache_key)
            if cached:
                logger.debug(f"Cache hit for key: {cache_key}")
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Cache retrieval error: {e}")
        
        return None
    
    def _cache_response(self, cache_key: str, response: Dict) -> None:
        """Cache response for future use."""
        if not self.cache_client:
            return
        
        try:
            self.cache_client.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(response)
            )
            logger.debug(f"Response cached with key: {cache_key}")
        except Exception as e:
            logger.warning(f"Cache storage error: {e}")
    
    def _ensure_model_available(self) -> None:
        """Ensure the configured model is available, pull if necessary."""
        try:
            # Check if model exists
            models = self.client.list()
            model_names = [model['name'] for model in models.get('models', [])]
            
            if self.config['model'] not in model_names:
                logger.info(f"Model {self.config['model']} not found, attempting to pull...")
                self.client.pull(self.config['model'])
                logger.info(f"Successfully pulled model {self.config['model']}")
                
        except Exception as e:
            logger.warning(f"Could not ensure model availability: {e}")
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_cache: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate chat completion with caching and error handling.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate (not directly supported by Ollama)
            use_cache: Whether to use response caching
            **kwargs: Additional parameters for Ollama API
        
        Returns:
            Dictionary containing response content and metadata
        
        Raises:
            LLMError: If API call fails
        """
        # Set defaults (Ollama doesn't use temperature the same way)
        temperature = temperature or 0.1
        
        # Generate cache key
        cache_key = self._generate_cache_key(
            messages, temperature=temperature, **kwargs
        )
        
        # Check cache first
        if use_cache:
            cached_response = self._get_cached_response(cache_key)
            if cached_response:
                return cached_response
        
        try:
            # Ensure model is available
            self._ensure_model_available()
            
            # Prepare options for Ollama
            options = {
                "temperature": temperature,
                **kwargs
            }
            
            # Make API call
            response = self.client.chat(
                model=self.config["model"],
                messages=messages,
                options=options
            )
            
            # Extract response data (Ollama format is different from OpenAI)
            result = {
                "content": response['message']['content'],
                "model": response.get('model', self.config["model"]),
                "usage": {
                    "prompt_tokens": response.get('prompt_eval_count', 0),
                    "completion_tokens": response.get('eval_count', 0),
                    "total_tokens": response.get('prompt_eval_count', 0) + response.get('eval_count', 0),
                },
                "created_at": get_utc_now().isoformat(),
                "finish_reason": "stop",  # Ollama doesn't provide finish_reason
                "eval_duration": response.get('eval_duration', 0),
                "load_duration": response.get('load_duration', 0),
            }
            
            # Cache the response
            if use_cache:
                self._cache_response(cache_key, result)
            
            logger.info(f"Ollama completion generated: {result['usage']['total_tokens']} tokens")
            return result
            
        except ResponseError as e:
            logger.error(f"Ollama API error: {e}")
            if e.status_code == 404:
                raise LLMError(f"Model {self.config['model']} not found. Try pulling it first.")
            raise LLMError(f"Ollama API call failed: {str(e)}")
        except Exception as e:
            logger.error(f"Ollama client error: {e}")
            raise LLMError(f"Ollama client error: {str(e)}")
    
    async def async_chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_cache: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Asynchronous chat completion with caching and error handling.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate (not directly supported by Ollama)
            use_cache: Whether to use response caching
            **kwargs: Additional parameters for Ollama API
        
        Returns:
            Dictionary containing response content and metadata
        
        Raises:
            LLMError: If API call fails
        """
        # Set defaults
        temperature = temperature or 0.1
        
        # Generate cache key
        cache_key = self._generate_cache_key(
            messages, temperature=temperature, **kwargs
        )
        
        # Check cache first
        if use_cache:
            cached_response = self._get_cached_response(cache_key)
            if cached_response:
                return cached_response
        
        try:
            # Prepare options for Ollama
            options = {
                "temperature": temperature,
                **kwargs
            }
            
            # Make async API call
            response = await self.async_client.chat(
                model=self.config["model"],
                messages=messages,
                options=options
            )
            
            # Extract response data
            result = {
                "content": response['message']['content'],
                "model": response.get('model', self.config["model"]),
                "usage": {
                    "prompt_tokens": response.get('prompt_eval_count', 0),
                    "completion_tokens": response.get('eval_count', 0),
                    "total_tokens": response.get('prompt_eval_count', 0) + response.get('eval_count', 0),
                },
                "created_at": get_utc_now().isoformat(),
                "finish_reason": "stop",
                "eval_duration": response.get('eval_duration', 0),
                "load_duration": response.get('load_duration', 0),
            }
            
            # Cache the response
            if use_cache:
                self._cache_response(cache_key, result)
            
            logger.info(f"Ollama async completion generated: {result['usage']['total_tokens']} tokens")
            return result
            
        except ResponseError as e:
            logger.error(f"Ollama async API error: {e}")
            if e.status_code == 404:
                raise LLMError(f"Model {self.config['model']} not found. Try pulling it first.")
            raise LLMError(f"Ollama async API call failed: {str(e)}")
        except Exception as e:
            logger.error(f"Ollama async client error: {e}")
            raise LLMError(f"Ollama async client error: {str(e)}")
    
    def stream_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Stream chat completion for real-time responses.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate (not directly supported by Ollama)
            **kwargs: Additional parameters for Ollama API
        
        Yields:
            Streaming response chunks
        
        Raises:
            LLMError: If API call fails
        """
        # Set defaults
        temperature = temperature or 0.1
        
        try:
            # Ensure model is available
            self._ensure_model_available()
            
            # Prepare options for Ollama
            options = {
                "temperature": temperature,
                **kwargs
            }
            
            stream = self.client.chat(
                model=self.config["model"],
                messages=messages,
                options=options,
                stream=True
            )
            
            for chunk in stream:
                if chunk.get('message', {}).get('content'):
                    yield chunk['message']['content']
                    
        except ResponseError as e:
            logger.error(f"Ollama streaming error: {e}")
            if e.status_code == 404:
                raise LLMError(f"Model {self.config['model']} not found. Try pulling it first.")
            raise LLMError(f"Ollama streaming failed: {str(e)}")
        except Exception as e:
            logger.error(f"Ollama streaming error: {e}")
            raise LLMError(f"Ollama streaming failed: {str(e)}")
    
    def test_connection(self) -> bool:
        """
        Test Ollama API connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Test with a simple list call first
            models = self.client.list()
            
            # Then test with a simple chat
            response = self.chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                use_cache=False
            )
            logger.info("Ollama connection test successful")
            return True
        except Exception as e:
            logger.error(f"Ollama connection test failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the configured model.
        
        Returns:
            Dictionary with model information
        """
        try:
            # Get model details from Ollama
            model_info = self.client.show(self.config["model"])
            return {
                "provider": "ollama",
                "model": self.config["model"],
                "host": self.config["host"],
                "timeout": self.config.get("timeout", 60.0),
                "details": model_info,
            }
        except Exception as e:
            logger.warning(f"Could not get model info: {e}")
            return {
                "provider": "ollama",
                "model": self.config["model"],
                "host": self.config["host"],
                "timeout": self.config.get("timeout", 60.0),
                "error": str(e),
            }
    
    def pull_model(self, model_name: Optional[str] = None) -> bool:
        """
        Pull a model from Ollama registry.
        
        Args:
            model_name: Name of model to pull (defaults to configured model)
        
        Returns:
            True if successful, False otherwise
        """
        model_name = model_name or self.config["model"]
        try:
            logger.info(f"Pulling Ollama model: {model_name}")
            self.client.pull(model_name)
            logger.info(f"Successfully pulled model: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List available Ollama models.
        
        Returns:
            List of model information dictionaries
        """
        try:
            response = self.client.list()
            return response.get('models', [])
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []


# Global client instance
_ollama_client: Optional[OllamaClient] = None


def get_ollama_client() -> OllamaClient:
    """Get or create Ollama client instance."""
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OllamaClient()
    return _ollama_client


def is_ollama_available() -> bool:
    """Check if Ollama is available and configured."""
    try:
        client = get_ollama_client()
        return client.test_connection()
    except Exception:
        return False 