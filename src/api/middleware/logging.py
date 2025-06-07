"""
Logging middleware for RiskFlow Credit Risk API.
Provides comprehensive request/response logging and monitoring.
"""

import time
import json
import uuid
from typing import Dict, Any, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import StreamingResponse

from ...config.logging_config import get_logger
from ...utils.database import db_manager

logger = get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive logging middleware for API requests and responses.
    Logs request details, response metrics, and performance data.
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.excluded_paths = {"/health", "/health/", "/docs", "/redoc", "/openapi.json"}
        self.sensitive_headers = {"authorization", "x-api-key", "cookie"}
        
    async def dispatch(self, request: Request, call_next):
        """Process request with comprehensive logging."""
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        # Log incoming request
        await self._log_request_start(request, request_id)
        
        try:
            # Process the request
            response = await call_next(request)
            
            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000
            
            # Add response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{response_time_ms:.2f}ms"
            
            # Log response
            await self._log_request_complete(
                request, 
                response, 
                request_id, 
                response_time_ms
            )
            
            return response
            
        except Exception as e:
            # Log error
            response_time_ms = (time.time() - start_time) * 1000
            await self._log_request_error(request, request_id, response_time_ms, str(e))
            raise
    
    async def _log_request_start(self, request: Request, request_id: str):
        """Log incoming request details."""
        # Skip logging for excluded paths
        if request.url.path in self.excluded_paths:
            return
        
        # Extract request details
        request_data = {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "headers": self._sanitize_headers(dict(request.headers)),
            "client_ip": self._get_client_ip(request),
            "user_agent": request.headers.get("user-agent", "unknown"),
            "timestamp": time.time(),
            "event": "request_start"
        }
        
        # Log body size for POST/PUT requests
        if request.method in ["POST", "PUT", "PATCH"]:
            content_length = request.headers.get("content-length")
            if content_length:
                request_data["content_length"] = int(content_length)
        
        logger.info(f"Request started: {request.method} {request.url.path}", extra=request_data)
    
    async def _log_request_complete(
        self, 
        request: Request, 
        response: Response, 
        request_id: str,
        response_time_ms: float
    ):
        """Log completed request with response details."""
        # Skip logging for excluded paths
        if request.url.path in self.excluded_paths:
            return
        
        # Extract response details
        response_data = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "response_time_ms": response_time_ms,
            "response_size": self._get_response_size(response),
            "client_ip": self._get_client_ip(request),
            "timestamp": time.time(),
            "event": "request_complete"
        }
        
        # Determine log level based on status code
        if response.status_code >= 500:
            log_level = "error"
        elif response.status_code >= 400:
            log_level = "warning"
        else:
            log_level = "info"
        
        # Log the response
        log_message = f"Request completed: {request.method} {request.url.path} - {response.status_code} ({response_time_ms:.2f}ms)"
        
        if log_level == "error":
            logger.error(log_message, extra=response_data)
        elif log_level == "warning":
            logger.warning(log_message, extra=response_data)
        else:
            logger.info(log_message, extra=response_data)
        
        # Store metrics in database for monitoring
        await self._store_request_metrics(request, response, request_id, response_time_ms)
        
        # Log slow requests
        if response_time_ms > 1000:  # > 1 second
            logger.warning(
                f"Slow request detected: {request.method} {request.url.path} took {response_time_ms:.2f}ms",
                extra={**response_data, "alert": "slow_request"}
            )
    
    async def _log_request_error(
        self, 
        request: Request, 
        request_id: str, 
        response_time_ms: float,
        error_message: str
    ):
        """Log request that resulted in an error."""
        error_data = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "error": error_message,
            "response_time_ms": response_time_ms,
            "client_ip": self._get_client_ip(request),
            "timestamp": time.time(),
            "event": "request_error"
        }
        
        logger.error(
            f"Request error: {request.method} {request.url.path} - {error_message}",
            extra=error_data
        )
        
        # Store error metrics
        try:
            db_manager.log_api_usage(
                request_id=request_id,
                endpoint=request.url.path,
                success=False,
                response_time_ms=response_time_ms,
                error_message=error_message
            )
        except Exception as e:
            logger.error(f"Failed to store error metrics: {str(e)}")
    
    def _sanitize_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Remove sensitive information from headers."""
        sanitized = {}
        for key, value in headers.items():
            if key.lower() in self.sensitive_headers:
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = value
        return sanitized
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check X-Forwarded-For header (proxy/load balancer)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            # Get first IP in case of proxy chain
            return forwarded_for.split(",")[0].strip()
        
        # Check X-Real-IP header (nginx)
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fall back to direct client IP
        return request.client.host if request.client else "unknown"
    
    def _get_response_size(self, response: Response) -> Optional[int]:
        """Get response size in bytes."""
        try:
            if hasattr(response, 'headers') and 'content-length' in response.headers:
                return int(response.headers['content-length'])
            
            # For streaming responses, we can't easily get the size
            if isinstance(response, StreamingResponse):
                return None
            
            # Try to get body size
            if hasattr(response, 'body') and response.body:
                return len(response.body)
            
        except Exception:
            pass
        
        return None
    
    async def _store_request_metrics(
        self, 
        request: Request, 
        response: Response, 
        request_id: str,
        response_time_ms: float
    ):
        """Store request metrics in database for monitoring."""
        try:
            # Determine if this was a successful request
            success = 200 <= response.status_code < 400
            
            # Extract batch size for batch requests
            batch_size = None
            if "/batch" in request.url.path and request.method == "POST":
                # Would need to parse request body to get actual batch size
                # For now, we'll extract it from response if available
                pass
            
            # Store in database
            db_manager.log_api_usage(
                request_id=request_id,
                endpoint=request.url.path,
                success=success,
                response_time_ms=response_time_ms,
                error_message=None if success else f"HTTP {response.status_code}",
                batch_size=batch_size
            )
            
        except Exception as e:
            # Don't let metrics storage failure affect the request
            logger.error(f"Failed to store request metrics: {str(e)}")


class PerformanceLoggingMiddleware(BaseHTTPMiddleware):
    """
    Specialized middleware for performance monitoring and alerting.
    Tracks detailed performance metrics and triggers alerts.
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.performance_thresholds = {
            "slow_request_ms": 1000,
            "very_slow_request_ms": 5000,
            "high_memory_mb": 500,
            "error_rate_threshold": 0.05  # 5%
        }
        
        # Performance tracking
        self.request_times = []
        self.error_count = 0
        self.total_requests = 0
        
    async def dispatch(self, request: Request, call_next):
        """Process request with performance monitoring."""
        start_time = time.time()
        memory_before = self._get_memory_usage()
        
        try:
            response = await call_next(request)
            
            # Calculate metrics
            response_time_ms = (time.time() - start_time) * 1000
            memory_after = self._get_memory_usage()
            memory_delta = memory_after - memory_before if memory_after and memory_before else None
            
            # Track performance
            self._track_performance(response_time_ms, response.status_code >= 400)
            
            # Check for performance alerts
            await self._check_performance_alerts(
                request, 
                response, 
                response_time_ms, 
                memory_delta
            )
            
            return response
            
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            self._track_performance(response_time_ms, True)
            
            # Log performance impact of errors
            logger.error(
                f"Request error impacted performance: {response_time_ms:.2f}ms",
                extra={
                    "path": request.url.path,
                    "error": str(e),
                    "response_time_ms": response_time_ms
                }
            )
            raise
    
    def _track_performance(self, response_time_ms: float, is_error: bool):
        """Track basic performance metrics."""
        self.request_times.append(response_time_ms)
        self.total_requests += 1
        
        if is_error:
            self.error_count += 1
        
        # Keep only recent requests (last 1000)
        if len(self.request_times) > 1000:
            self.request_times.pop(0)
    
    async def _check_performance_alerts(
        self, 
        request: Request, 
        response: Response,
        response_time_ms: float,
        memory_delta: Optional[float]
    ):
        """Check for performance issues and trigger alerts."""
        alerts = []
        
        # Check slow requests
        if response_time_ms > self.performance_thresholds["very_slow_request_ms"]:
            alerts.append({
                "type": "very_slow_request",
                "threshold": self.performance_thresholds["very_slow_request_ms"],
                "actual": response_time_ms,
                "severity": "high"
            })
        elif response_time_ms > self.performance_thresholds["slow_request_ms"]:
            alerts.append({
                "type": "slow_request",
                "threshold": self.performance_thresholds["slow_request_ms"],
                "actual": response_time_ms,
                "severity": "medium"
            })
        
        # Check memory usage
        if memory_delta and memory_delta > self.performance_thresholds["high_memory_mb"]:
            alerts.append({
                "type": "high_memory_usage",
                "threshold": self.performance_thresholds["high_memory_mb"],
                "actual": memory_delta,
                "severity": "medium"
            })
        
        # Check error rate
        if self.total_requests > 10:  # Only check after some requests
            error_rate = self.error_count / self.total_requests
            if error_rate > self.performance_thresholds["error_rate_threshold"]:
                alerts.append({
                    "type": "high_error_rate",
                    "threshold": self.performance_thresholds["error_rate_threshold"],
                    "actual": error_rate,
                    "severity": "high"
                })
        
        # Log alerts
        for alert in alerts:
            logger.warning(
                f"Performance alert: {alert['type']} - {alert['actual']} exceeds {alert['threshold']}",
                extra={
                    "alert": alert,
                    "path": request.url.path,
                    "method": request.method,
                    "status_code": response.status_code
                }
            )
    
    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            # psutil not available
            return None
        except Exception:
            return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        if not self.request_times:
            return {"status": "no_data"}
        
        request_times = self.request_times[-100:]  # Last 100 requests
        
        return {
            "total_requests": self.total_requests,
            "error_count": self.error_count,
            "error_rate": self.error_count / self.total_requests if self.total_requests > 0 else 0,
            "avg_response_time_ms": sum(request_times) / len(request_times),
            "min_response_time_ms": min(request_times),
            "max_response_time_ms": max(request_times),
            "p95_response_time_ms": self._percentile(request_times, 95),
            "p99_response_time_ms": self._percentile(request_times, 99),
            "slow_requests": len([t for t in request_times if t > self.performance_thresholds["slow_request_ms"]]),
            "very_slow_requests": len([t for t in request_times if t > self.performance_thresholds["very_slow_request_ms"]])
        }
    
    def _percentile(self, data: list, percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]


# Global performance monitor instance
performance_monitor = None


def get_performance_monitor() -> Optional[PerformanceLoggingMiddleware]:
    """Get global performance monitor instance."""
    return performance_monitor


def set_performance_monitor(monitor: PerformanceLoggingMiddleware):
    """Set global performance monitor instance."""
    global performance_monitor
    performance_monitor = monitor