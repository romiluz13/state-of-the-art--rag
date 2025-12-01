"""Middleware for request logging, correlation IDs, and metrics."""

import logging
import time
import uuid
from contextvars import ContextVar
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

# Context variable for request correlation ID
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")


def get_correlation_id() -> str:
    """Get current request's correlation ID."""
    return correlation_id_var.get()


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """Add correlation ID to each request for tracing."""

    HEADER_NAME = "X-Correlation-ID"

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add correlation ID to request and response."""
        # Get from header or generate new
        correlation_id = request.headers.get(
            self.HEADER_NAME,
            str(uuid.uuid4())[:8]
        )

        # Set in context
        token = correlation_id_var.set(correlation_id)

        try:
            response = await call_next(request)
            # Add to response headers
            response.headers[self.HEADER_NAME] = correlation_id
            return response
        finally:
            correlation_id_var.reset(token)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests with timing and status."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log request start, end, and duration."""
        start_time = time.perf_counter()
        correlation_id = get_correlation_id()

        # Log request start
        logger.info(
            f"[{correlation_id}] {request.method} {request.url.path} - Started",
            extra={
                "correlation_id": correlation_id,
                "method": request.method,
                "path": request.url.path,
                "query": str(request.query_params),
            }
        )

        try:
            response = await call_next(request)
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Log request completion
            logger.info(
                f"[{correlation_id}] {request.method} {request.url.path} - "
                f"{response.status_code} ({duration_ms:.2f}ms)",
                extra={
                    "correlation_id": correlation_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": duration_ms,
                }
            )

            # Add timing header
            response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
            return response

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"[{correlation_id}] {request.method} {request.url.path} - "
                f"Error ({duration_ms:.2f}ms): {str(e)}",
                extra={
                    "correlation_id": correlation_id,
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": duration_ms,
                    "error": str(e),
                },
                exc_info=True,
            )
            raise


class MetricsCollector:
    """Collect and expose application metrics."""

    def __init__(self):
        """Initialize metrics collector."""
        self._request_count: dict[str, int] = {}
        self._request_latencies: dict[str, list[float]] = {}
        self._error_count: dict[str, int] = {}
        self._strategy_usage: dict[str, int] = {}

    def record_request(
        self,
        path: str,
        method: str,
        status_code: int,
        duration_ms: float,
    ) -> None:
        """Record a request metric.

        Args:
            path: Request path
            method: HTTP method
            status_code: Response status code
            duration_ms: Request duration in milliseconds
        """
        key = f"{method}:{path}"

        # Count requests
        self._request_count[key] = self._request_count.get(key, 0) + 1

        # Track latencies (keep last 100)
        if key not in self._request_latencies:
            self._request_latencies[key] = []
        latencies = self._request_latencies[key]
        latencies.append(duration_ms)
        if len(latencies) > 100:
            self._request_latencies[key] = latencies[-100:]

        # Count errors
        if status_code >= 400:
            error_key = f"{key}:{status_code}"
            self._error_count[error_key] = self._error_count.get(error_key, 0) + 1

    def record_strategy_usage(self, strategy: str) -> None:
        """Record strategy usage.

        Args:
            strategy: Strategy name used for retrieval
        """
        self._strategy_usage[strategy] = self._strategy_usage.get(strategy, 0) + 1

    def get_metrics(self) -> dict:
        """Get all collected metrics."""
        metrics = {
            "requests": {
                "total": sum(self._request_count.values()),
                "by_endpoint": self._request_count.copy(),
            },
            "latencies": {},
            "errors": {
                "total": sum(self._error_count.values()),
                "by_endpoint": self._error_count.copy(),
            },
            "strategies": self._strategy_usage.copy(),
        }

        # Calculate latency percentiles
        for key, latencies in self._request_latencies.items():
            if latencies:
                sorted_lat = sorted(latencies)
                n = len(sorted_lat)
                metrics["latencies"][key] = {
                    "count": n,
                    "avg_ms": sum(latencies) / n,
                    "p50_ms": sorted_lat[n // 2],
                    "p95_ms": sorted_lat[int(n * 0.95)] if n >= 20 else sorted_lat[-1],
                    "p99_ms": sorted_lat[int(n * 0.99)] if n >= 100 else sorted_lat[-1],
                }

        return metrics

    def reset(self) -> None:
        """Reset all metrics."""
        self._request_count.clear()
        self._request_latencies.clear()
        self._error_count.clear()
        self._strategy_usage.clear()


# Global metrics collector
_metrics_collector: MetricsCollector | None = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


class MetricsMiddleware(BaseHTTPMiddleware):
    """Collect request metrics."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Record request metrics."""
        start_time = time.perf_counter()

        response = await call_next(request)

        duration_ms = (time.perf_counter() - start_time) * 1000
        collector = get_metrics_collector()
        collector.record_request(
            path=request.url.path,
            method=request.method,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )

        return response
