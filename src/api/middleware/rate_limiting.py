"""
Rate limiting middleware for RiskFlow Credit Risk API.
Implements request rate limiting and quota management.
"""

import time
import hashlib
from typing import Dict, Optional
from collections import defaultdict, deque
from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ...config.settings import get_settings
from ...config.logging_config import get_logger

logger = get_logger(__name__)
settings = get_settings()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware using sliding window algorithm.
    Tracks requests per client and enforces rate limits.
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.rate_limits = {
            "default": {"requests": 100, "window": 60},  # 100 requests per minute
            "predictions": {"requests": 50, "window": 60},  # 50 predictions per minute
            "batch": {"requests": 10, "window": 300},  # 10 batch requests per 5 minutes
        }
        
        # In-memory storage (would use Redis in production)
        self.client_requests = defaultdict(lambda: defaultdict(deque))
        self.client_quotas = {}  # For API key-based quotas
        
    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting."""
        start_time = time.time()
        
        try:
            # Get client identifier
            client_id = self._get_client_id(request)
            
            # Determine rate limit based on endpoint
            rate_limit = self._get_rate_limit(request.url.path)
            
            # Check if request is allowed
            allowed, remaining, reset_time = self._is_request_allowed(
                client_id, 
                rate_limit["requests"], 
                rate_limit["window"]
            )
            
            if not allowed:
                logger.warning(f"Rate limit exceeded for client {client_id} on {request.url.path}")
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": "Rate limit exceeded",
                        "message": f"Too many requests. Limit: {rate_limit['requests']} per {rate_limit['window']} seconds",
                        "retry_after": int(reset_time - time.time()),
                        "remaining_requests": 0,
                        "reset_time": int(reset_time),
                        "client_id": client_id
                    },
                    headers={
                        "X-RateLimit-Limit": str(rate_limit["requests"]),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(int(reset_time)),
                        "Retry-After": str(int(reset_time - time.time()))
                    }
                )
            
            # Record the request
            self._record_request(client_id, rate_limit["window"])
            
            # Process the request
            response = await call_next(request)
            
            # Add rate limit headers to response
            response.headers["X-RateLimit-Limit"] = str(rate_limit["requests"])
            response.headers["X-RateLimit-Remaining"] = str(max(0, remaining - 1))
            response.headers["X-RateLimit-Reset"] = str(int(reset_time))
            response.headers["X-Request-ID"] = client_id
            
            # Log request metrics
            response_time = (time.time() - start_time) * 1000
            self._log_request_metrics(request, response, response_time, client_id)
            
            return response
            
        except Exception as e:
            logger.error(f"Rate limiting middleware error: {str(e)}")
            # Don't block requests due to rate limiting errors
            response = await call_next(request)
            return response
    
    def _get_client_id(self, request: Request) -> str:
        """Extract client identifier from request."""
        # Try API key first
        api_key = request.headers.get("X-API-Key") or request.headers.get("Authorization")
        if api_key:
            return hashlib.md5(api_key.encode()).hexdigest()[:16]
        
        # Fall back to IP address
        client_ip = request.headers.get("X-Forwarded-For")
        if client_ip:
            # Get first IP in case of proxy chain
            client_ip = client_ip.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "unknown"
        
        return f"ip_{client_ip}"
    
    def _get_rate_limit(self, path: str) -> Dict[str, int]:
        """Determine rate limit based on endpoint path."""
        if "/predictions/batch" in path:
            return self.rate_limits["batch"]
        elif "/predictions" in path:
            return self.rate_limits["predictions"]
        else:
            return self.rate_limits["default"]
    
    def _is_request_allowed(
        self, 
        client_id: str, 
        max_requests: int, 
        window_seconds: int
    ) -> tuple[bool, int, float]:
        """
        Check if request is allowed using sliding window algorithm.
        
        Returns:
            - allowed: Whether request is allowed
            - remaining: Number of remaining requests
            - reset_time: When the window resets
        """
        current_time = time.time()
        window_start = current_time - window_seconds
        
        # Get client's request history
        request_times = self.client_requests[client_id]["requests"]
        
        # Remove old requests outside the window
        while request_times and request_times[0] < window_start:
            request_times.popleft()
        
        # Check if under limit
        current_requests = len(request_times)
        allowed = current_requests < max_requests
        remaining = max(0, max_requests - current_requests)
        
        # Calculate reset time (when oldest request will expire)
        if request_times:
            reset_time = request_times[0] + window_seconds
        else:
            reset_time = current_time + window_seconds
        
        return allowed, remaining, reset_time
    
    def _record_request(self, client_id: str, window_seconds: int):
        """Record a request for rate limiting tracking."""
        current_time = time.time()
        self.client_requests[client_id]["requests"].append(current_time)
        
        # Clean up old requests to prevent memory bloat
        window_start = current_time - window_seconds
        request_times = self.client_requests[client_id]["requests"]
        while request_times and request_times[0] < window_start:
            request_times.popleft()
    
    def _log_request_metrics(
        self, 
        request: Request, 
        response: Response, 
        response_time_ms: float,
        client_id: str
    ):
        """Log request metrics for monitoring."""
        try:
            metrics = {
                "client_id": client_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "response_time_ms": response_time_ms,
                "user_agent": request.headers.get("User-Agent", "unknown"),
                "timestamp": time.time()
            }
            
            # Log slow requests
            if response_time_ms > 1000:  # > 1 second
                logger.warning(f"Slow request detected: {metrics}")
            
            # Log errors
            if response.status_code >= 400:
                logger.info(f"Error response: {metrics}")
            
        except Exception as e:
            logger.error(f"Failed to log request metrics: {str(e)}")
    
    def get_client_stats(self, client_id: str) -> Dict[str, any]:
        """Get current statistics for a client."""
        current_time = time.time()
        stats = {}
        
        for rate_type, config in self.rate_limits.items():
            window_start = current_time - config["window"]
            request_times = self.client_requests[client_id]["requests"]
            
            # Count requests in current window
            current_requests = sum(1 for t in request_times if t >= window_start)
            
            stats[rate_type] = {
                "current_requests": current_requests,
                "max_requests": config["requests"],
                "window_seconds": config["window"],
                "remaining": max(0, config["requests"] - current_requests)
            }
        
        return stats
    
    def reset_client_limits(self, client_id: str):
        """Reset rate limits for a specific client (admin function)."""
        if client_id in self.client_requests:
            del self.client_requests[client_id]
        logger.info(f"Rate limits reset for client: {client_id}")


class APIKeyRateLimitMiddleware(BaseHTTPMiddleware):
    """
    Enhanced rate limiting with API key-based quotas.
    Provides different rate limits based on API key tiers.
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.tier_limits = {
            "free": {"requests_per_hour": 100, "batch_size": 10},
            "standard": {"requests_per_hour": 1000, "batch_size": 50},
            "premium": {"requests_per_hour": 10000, "batch_size": 200},
            "enterprise": {"requests_per_hour": 100000, "batch_size": 1000}
        }
        
        # API key to tier mapping (would be in database in production)
        self.api_key_tiers = {}
        
        # Request tracking per API key
        self.api_key_requests = defaultdict(lambda: defaultdict(deque))
    
    async def dispatch(self, request: Request, call_next):
        """Process request with API key-based rate limiting."""
        # Extract API key
        api_key = self._extract_api_key(request)
        
        if api_key:
            # Get tier for API key
            tier = self._get_api_key_tier(api_key)
            
            # Check tier-specific limits
            allowed, reason = self._check_tier_limits(api_key, tier, request)
            
            if not allowed:
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": "API quota exceeded",
                        "message": reason,
                        "tier": tier,
                        "upgrade_info": "Contact support for higher limits"
                    }
                )
        
        # Continue with standard rate limiting
        return await call_next(request)
    
    def _extract_api_key(self, request: Request) -> Optional[str]:
        """Extract API key from request headers."""
        # Check X-API-Key header
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return api_key
        
        # Check Authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header[7:]  # Remove "Bearer " prefix
        
        return None
    
    def _get_api_key_tier(self, api_key: str) -> str:
        """Get tier for API key (would query database in production)."""
        return self.api_key_tiers.get(api_key, "free")
    
    def _check_tier_limits(self, api_key: str, tier: str, request: Request) -> tuple[bool, str]:
        """Check if request is within tier limits."""
        tier_config = self.tier_limits.get(tier, self.tier_limits["free"])
        current_time = time.time()
        hour_start = current_time - 3600  # 1 hour window
        
        # Check hourly request limit
        request_times = self.api_key_requests[api_key]["hourly"]
        
        # Remove old requests
        while request_times and request_times[0] < hour_start:
            request_times.popleft()
        
        current_hourly_requests = len(request_times)
        
        if current_hourly_requests >= tier_config["requests_per_hour"]:
            return False, f"Hourly limit exceeded for {tier} tier ({tier_config['requests_per_hour']} requests/hour)"
        
        # Check batch size limits for batch endpoints
        if "/batch" in request.url.path:
            # Would need to parse request body to check batch size
            # This is a simplified check
            pass
        
        # Record the request
        request_times.append(current_time)
        
        return True, "OK"


# Utility functions for rate limit management
def get_rate_limit_stats() -> Dict[str, any]:
    """Get overall rate limiting statistics."""
    # Would implement comprehensive stats collection
    return {
        "total_requests": 0,
        "rate_limited_requests": 0,
        "active_clients": 0,
        "average_response_time": 0
    }


def configure_custom_rate_limits(limits: Dict[str, Dict[str, int]]):
    """Configure custom rate limits (admin function)."""
    # Would update rate limits dynamically
    logger.info(f"Custom rate limits configured: {limits}")