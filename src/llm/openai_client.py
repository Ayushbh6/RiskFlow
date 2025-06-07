"""
OpenAI API client for credit risk analysis and documentation generation.
"""

import os
import json
import hashlib
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import asyncio
from pathlib import Path

from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion
import httpx

from ..config.settings import get_openai_config, settings
from ..utils.cache import get_cache_client
from ..utils.exceptions import LLMError, ConfigurationException
from ..utils.helpers import get_utc_now

logger = logging.getLogger(__name__)


class OpenAIClient:
    """OpenAI API client with caching, error handling, and retry logic."""
    
    def __init__(self):
        """Initialize OpenAI client with configuration."""
        self.config = get_openai_config()
        self.cache_client = get_cache_client()
        self.cache_ttl = 3600  # 1 hour cache TTL
        
        if not self.config.get("api_key"):
            raise ConfigurationException("OpenAI API key not provided")
        
        # Initialize synchronous client
        self.client = OpenAI(
            api_key=self.config["api_key"],
            timeout=self.config.get("timeout", 30.0),
            max_retries=3,
        )
        
        # Initialize asynchronous client
        self.async_client = AsyncOpenAI(
            api_key=self.config["api_key"],
            timeout=self.config.get("timeout", 30.0),
            max_retries=3,
        )
        
        logger.info(f"OpenAI client initialized with model: {self.config['model']}")
    
    def _generate_cache_key(self, messages: List[Dict], **kwargs) -> str:
        """Generate cache key for request."""
        cache_data = {
            "messages": messages,
            "model": self.config["model"],
            **kwargs
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return f"openai:{hashlib.md5(cache_string.encode()).hexdigest()}"
    
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
            max_tokens: Maximum tokens to generate
            use_cache: Whether to use response caching
            **kwargs: Additional parameters for OpenAI API
        
        Returns:
            Dictionary containing response content and metadata
        
        Raises:
            LLMError: If API call fails
        """
        # Set defaults
        temperature = temperature or self.config.get("temperature", 0.1)
        max_tokens = max_tokens or self.config.get("max_tokens", 2000)
        
        # Generate cache key
        cache_key = self._generate_cache_key(
            messages, temperature=temperature, max_tokens=max_tokens, **kwargs
        )
        
        # Check cache first
        if use_cache:
            cached_response = self._get_cached_response(cache_key)
            if cached_response:
                return cached_response
        
        try:
            # Make API call
            response = self.client.chat.completions.create(
                model=self.config["model"],
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            # Extract response data
            result = {
                "content": response.choices[0].message.content,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                "created_at": get_utc_now().isoformat(),
                "finish_reason": response.choices[0].finish_reason,
            }
            
            # Cache the response
            if use_cache:
                self._cache_response(cache_key, result)
            
            logger.info(f"OpenAI completion generated: {result['usage']['total_tokens']} tokens")
            return result
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise LLMError(f"OpenAI API call failed: {str(e)}")
    
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
            max_tokens: Maximum tokens to generate
            use_cache: Whether to use response caching
            **kwargs: Additional parameters for OpenAI API
        
        Returns:
            Dictionary containing response content and metadata
        
        Raises:
            LLMError: If API call fails
        """
        # Set defaults
        temperature = temperature or self.config.get("temperature", 0.1)
        max_tokens = max_tokens or self.config.get("max_tokens", 2000)
        
        # Generate cache key
        cache_key = self._generate_cache_key(
            messages, temperature=temperature, max_tokens=max_tokens, **kwargs
        )
        
        # Check cache first
        if use_cache:
            cached_response = self._get_cached_response(cache_key)
            if cached_response:
                return cached_response
        
        try:
            # Make async API call
            response = await self.async_client.chat.completions.create(
                model=self.config["model"],
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            # Extract response data
            result = {
                "content": response.choices[0].message.content,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                "created_at": get_utc_now().isoformat(),
                "finish_reason": response.choices[0].finish_reason,
            }
            
            # Cache the response
            if use_cache:
                self._cache_response(cache_key, result)
            
            logger.info(f"OpenAI async completion generated: {result['usage']['total_tokens']} tokens")
            return result
            
        except Exception as e:
            logger.error(f"OpenAI async API error: {e}")
            raise LLMError(f"OpenAI async API call failed: {str(e)}")
    
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
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters for OpenAI API
        
        Yields:
            Streaming response chunks
        
        Raises:
            LLMError: If API call fails
        """
        # Set defaults
        temperature = temperature or self.config.get("temperature", 0.1)
        max_tokens = max_tokens or self.config.get("max_tokens", 2000)
        
        try:
            stream = self.client.chat.completions.create(
                model=self.config["model"],
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise LLMError(f"OpenAI streaming failed: {str(e)}")
    
    def test_connection(self) -> bool:
        """
        Test OpenAI API connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = self.chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10,
                use_cache=False
            )
            logger.info("OpenAI connection test successful")
            return True
        except Exception as e:
            logger.error(f"OpenAI connection test failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the configured model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "provider": "openai",
            "model": self.config["model"],
            "max_tokens": self.config.get("max_tokens", 2000),
            "temperature": self.config.get("temperature", 0.1),
            "timeout": self.config.get("timeout", 30.0),
        }


# Global client instance
_openai_client: Optional[OpenAIClient] = None


def get_openai_client() -> OpenAIClient:
    """Get or create OpenAI client instance."""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAIClient()
    return _openai_client


def is_openai_available() -> bool:
    """Check if OpenAI is available and configured."""
    try:
        config = get_openai_config()
        return bool(config.get("api_key"))
    except Exception:
        return False 