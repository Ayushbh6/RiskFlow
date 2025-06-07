"""
Cache utilities for the credit risk MLOps system.
Provides Redis-based caching for LLM responses and other data.
"""

import json
import hashlib
import logging
from typing import Any, Optional, Dict, Union
from datetime import datetime, timedelta

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from config.settings import get_settings

logger = logging.getLogger(__name__)


class CacheClient:
    """Redis-based cache client with fallback to in-memory cache."""
    
    def __init__(self):
        self.settings = get_settings()
        self.redis_client = None
        self.memory_cache = {}  # Fallback in-memory cache
        
        if REDIS_AVAILABLE and hasattr(self.settings, 'redis_url'):
            try:
                self.redis_client = redis.from_url(
                    self.settings.redis_url,
                    decode_responses=True,
                    socket_timeout=5,
                    socket_connect_timeout=5
                )
                # Test connection
                self.redis_client.ping()
                logger.info("Connected to Redis cache")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}. Using in-memory cache.")
                self.redis_client = None
        else:
            logger.info("Redis not available. Using in-memory cache.")
    
    def _generate_key(self, prefix: str, data: Any) -> str:
        """Generate a cache key from data."""
        if isinstance(data, dict):
            # Sort dict for consistent hashing
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
        
        hash_obj = hashlib.md5(data_str.encode())
        return f"{prefix}:{hash_obj.hexdigest()}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            if self.redis_client:
                value = self.redis_client.get(key)
                if value:
                    return json.loads(value)
            else:
                # Check memory cache
                cache_entry = self.memory_cache.get(key)
                if cache_entry:
                    # Check expiration
                    if cache_entry.get('expires_at'):
                        if datetime.now() > datetime.fromisoformat(cache_entry['expires_at']):
                            del self.memory_cache[key]
                            return None
                    return cache_entry['value']
        except Exception as e:
            logger.error(f"Cache get error: {e}")
        
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache with TTL in seconds."""
        try:
            if self.redis_client:
                serialized = json.dumps(value)
                return self.redis_client.setex(key, ttl, serialized)
            else:
                # Store in memory cache with expiration
                expires_at = datetime.now() + timedelta(seconds=ttl)
                self.memory_cache[key] = {
                    'value': value,
                    'expires_at': expires_at.isoformat()
                }
                return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            if self.redis_client:
                return bool(self.redis_client.delete(key))
            else:
                if key in self.memory_cache:
                    del self.memory_cache[key]
                    return True
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
        
        return False
    
    def clear(self) -> bool:
        """Clear all cache entries."""
        try:
            if self.redis_client:
                return self.redis_client.flushdb()
            else:
                self.memory_cache.clear()
                return True
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "type": "redis" if self.redis_client else "memory",
            "connected": self.redis_client is not None
        }
        
        try:
            if self.redis_client:
                info = self.redis_client.info()
                stats.update({
                    "used_memory": info.get("used_memory_human", "N/A"),
                    "connected_clients": info.get("connected_clients", 0),
                    "total_commands_processed": info.get("total_commands_processed", 0)
                })
            else:
                stats.update({
                    "memory_entries": len(self.memory_cache),
                    "memory_size_bytes": sum(
                        len(str(entry)) for entry in self.memory_cache.values()
                    )
                })
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            stats["error"] = str(e)
        
        return stats


class LLMCache:
    """Specialized cache for LLM responses."""
    
    def __init__(self, cache_client: Optional[CacheClient] = None):
        self.cache_client = cache_client or get_cache_client()
        self.default_ttl = 3600 * 24  # 24 hours
    
    def get_response(self, messages: list, model: str, provider: str) -> Optional[Dict[str, Any]]:
        """Get cached LLM response."""
        cache_key = self._generate_response_key(messages, model, provider)
        cached = self.cache_client.get(cache_key)
        
        if cached:
            logger.debug(f"Cache hit for {provider}:{model}")
            cached["cached"] = True
            return cached
        
        return None
    
    def set_response(self, messages: list, model: str, provider: str, 
                    response: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Cache LLM response."""
        cache_key = self._generate_response_key(messages, model, provider)
        ttl = ttl or self.default_ttl
        
        # Add cache metadata
        cache_data = response.copy()
        cache_data.update({
            "cached_at": datetime.now().isoformat(),
            "cache_ttl": ttl
        })
        
        success = self.cache_client.set(cache_key, cache_data, ttl)
        if success:
            logger.debug(f"Cached response for {provider}:{model}")
        
        return success
    
    def _generate_response_key(self, messages: list, model: str, provider: str) -> str:
        """Generate cache key for LLM response."""
        cache_data = {
            "messages": messages,
            "model": model,
            "provider": provider
        }
        return self.cache_client._generate_key("llm_response", cache_data)
    
    def clear_provider_cache(self, provider: str) -> int:
        """Clear all cached responses for a provider."""
        # This is a simplified implementation
        # In production, you'd want to use Redis patterns or tags
        logger.info(f"Cache clear requested for provider: {provider}")
        return 0


# Global cache client instance
_cache_client = None


def get_cache_client() -> CacheClient:
    """Get global cache client instance."""
    global _cache_client
    if _cache_client is None:
        _cache_client = CacheClient()
    return _cache_client


def get_llm_cache() -> LLMCache:
    """Get LLM cache instance."""
    return LLMCache(get_cache_client())


def clear_all_cache() -> bool:
    """Clear all cache entries."""
    cache_client = get_cache_client()
    return cache_client.clear()


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    cache_client = get_cache_client()
    return cache_client.get_stats() 