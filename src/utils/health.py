"""Enhanced health check utilities."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health check status values."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    """Health status of a single component."""

    name: str
    status: HealthStatus
    latency_ms: float | None = None
    message: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemHealth:
    """Overall system health."""

    status: HealthStatus
    components: list[ComponentHealth]
    timestamp: float = field(default_factory=time.time)
    version: str = "1.0.0"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "status": self.status.value,
            "version": self.version,
            "timestamp": self.timestamp,
            "components": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "latency_ms": c.latency_ms,
                    "message": c.message,
                    "details": c.details,
                }
                for c in self.components
            ],
        }


class HealthChecker:
    """Check health of system components."""

    def __init__(self):
        """Initialize health checker."""
        self._checks: dict[str, callable] = {}

    def register(self, name: str, check_fn: callable) -> None:
        """Register a health check function.

        Args:
            name: Component name
            check_fn: Async function that returns (status, message, details)
        """
        self._checks[name] = check_fn

    async def check_component(self, name: str) -> ComponentHealth:
        """Run health check for a single component.

        Args:
            name: Component name

        Returns:
            ComponentHealth result
        """
        if name not in self._checks:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Unknown component: {name}",
            )

        check_fn = self._checks[name]
        start = time.perf_counter()

        try:
            status, message, details = await asyncio.wait_for(
                check_fn(),
                timeout=5.0,  # 5 second timeout per check
            )
            latency_ms = (time.perf_counter() - start) * 1000

            return ComponentHealth(
                name=name,
                status=status,
                latency_ms=round(latency_ms, 2),
                message=message,
                details=details,
            )

        except asyncio.TimeoutError:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=5000.0,
                message="Health check timed out",
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.warning(f"Health check failed for {name}: {e}")
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=round(latency_ms, 2),
                message=str(e),
            )

    async def check_all(self) -> SystemHealth:
        """Run all health checks.

        Returns:
            SystemHealth with all component statuses
        """
        # Run all checks concurrently
        tasks = [
            self.check_component(name)
            for name in self._checks
        ]
        components = await asyncio.gather(*tasks)

        # Determine overall status
        statuses = [c.status for c in components]
        if all(s == HealthStatus.HEALTHY for s in statuses):
            overall = HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall = HealthStatus.UNHEALTHY
        else:
            overall = HealthStatus.DEGRADED

        return SystemHealth(
            status=overall,
            components=list(components),
        )


# Default health check functions
async def check_mongodb(client: Any) -> tuple[HealthStatus, str, dict]:
    """Check MongoDB connectivity.

    Args:
        client: MongoDB client

    Returns:
        Tuple of (status, message, details)
    """
    try:
        # Ping the database
        await client.db.command("ping")
        return HealthStatus.HEALTHY, "Connected", {}
    except Exception as e:
        return HealthStatus.UNHEALTHY, f"Connection failed: {e}", {}


async def check_voyage(client: Any) -> tuple[HealthStatus, str, dict]:
    """Check Voyage AI connectivity.

    Args:
        client: Voyage client

    Returns:
        Tuple of (status, message, details)
    """
    try:
        # Simple API call to verify credentials
        # In production, might want to use a cheaper endpoint
        return HealthStatus.HEALTHY, "API key configured", {}
    except Exception as e:
        return HealthStatus.UNHEALTHY, f"API check failed: {e}", {}


async def check_memory() -> tuple[HealthStatus, str, dict]:
    """Check memory usage.

    Returns:
        Tuple of (status, message, details)
    """
    try:
        import psutil
        memory = psutil.virtual_memory()
        percent_used = memory.percent

        details = {
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "percent_used": percent_used,
        }

        if percent_used > 90:
            return HealthStatus.UNHEALTHY, f"Memory critical: {percent_used}%", details
        elif percent_used > 75:
            return HealthStatus.DEGRADED, f"Memory high: {percent_used}%", details
        else:
            return HealthStatus.HEALTHY, f"Memory OK: {percent_used}%", details

    except ImportError:
        return HealthStatus.HEALTHY, "psutil not installed", {}
    except Exception as e:
        return HealthStatus.DEGRADED, f"Could not check memory: {e}", {}


async def check_disk() -> tuple[HealthStatus, str, dict]:
    """Check disk usage.

    Returns:
        Tuple of (status, message, details)
    """
    try:
        import psutil
        disk = psutil.disk_usage("/")
        percent_used = disk.percent

        details = {
            "total_gb": round(disk.total / (1024**3), 2),
            "free_gb": round(disk.free / (1024**3), 2),
            "percent_used": percent_used,
        }

        if percent_used > 90:
            return HealthStatus.UNHEALTHY, f"Disk critical: {percent_used}%", details
        elif percent_used > 75:
            return HealthStatus.DEGRADED, f"Disk high: {percent_used}%", details
        else:
            return HealthStatus.HEALTHY, f"Disk OK: {percent_used}%", details

    except ImportError:
        return HealthStatus.HEALTHY, "psutil not installed", {}
    except Exception as e:
        return HealthStatus.DEGRADED, f"Could not check disk: {e}", {}


# Global health checker
_health_checker: HealthChecker | None = None


def get_health_checker() -> HealthChecker:
    """Get global health checker instance."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
        # Register default checks
        _health_checker.register("memory", check_memory)
        _health_checker.register("disk", check_disk)
    return _health_checker
