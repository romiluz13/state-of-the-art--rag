"""Tests for health check utilities."""

import pytest

from src.utils.health import (
    HealthStatus,
    ComponentHealth,
    SystemHealth,
    HealthChecker,
    check_memory,
    check_disk,
)


class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_status_values(self):
        """Test health status values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"


class TestComponentHealth:
    """Tests for ComponentHealth dataclass."""

    def test_basic_creation(self):
        """Test basic component health creation."""
        health = ComponentHealth(
            name="test",
            status=HealthStatus.HEALTHY,
        )
        assert health.name == "test"
        assert health.status == HealthStatus.HEALTHY
        assert health.latency_ms is None
        assert health.message is None

    def test_with_details(self):
        """Test component health with all fields."""
        health = ComponentHealth(
            name="mongodb",
            status=HealthStatus.HEALTHY,
            latency_ms=5.2,
            message="Connected",
            details={"connections": 10},
        )
        assert health.latency_ms == 5.2
        assert health.message == "Connected"
        assert health.details["connections"] == 10


class TestSystemHealth:
    """Tests for SystemHealth dataclass."""

    def test_basic_creation(self):
        """Test basic system health creation."""
        components = [
            ComponentHealth(name="a", status=HealthStatus.HEALTHY),
            ComponentHealth(name="b", status=HealthStatus.HEALTHY),
        ]
        health = SystemHealth(
            status=HealthStatus.HEALTHY,
            components=components,
        )
        assert health.status == HealthStatus.HEALTHY
        assert len(health.components) == 2

    def test_to_dict(self):
        """Test conversion to dictionary."""
        components = [
            ComponentHealth(
                name="test",
                status=HealthStatus.HEALTHY,
                latency_ms=5.0,
                message="OK",
            ),
        ]
        health = SystemHealth(
            status=HealthStatus.HEALTHY,
            components=components,
        )

        d = health.to_dict()
        assert d["status"] == "healthy"
        assert len(d["components"]) == 1
        assert d["components"][0]["name"] == "test"
        assert d["components"][0]["status"] == "healthy"


class TestHealthChecker:
    """Tests for HealthChecker."""

    @pytest.fixture
    def checker(self):
        """Create health checker."""
        return HealthChecker()

    @pytest.mark.asyncio
    async def test_register_check(self, checker):
        """Test registering a health check."""
        async def my_check():
            return HealthStatus.HEALTHY, "All good", {}

        checker.register("mycomponent", my_check)
        result = await checker.check_component("mycomponent")

        assert result.status == HealthStatus.HEALTHY
        assert result.message == "All good"

    @pytest.mark.asyncio
    async def test_unknown_component(self, checker):
        """Test checking unknown component."""
        result = await checker.check_component("unknown")
        assert result.status == HealthStatus.UNHEALTHY
        assert "Unknown component" in result.message

    @pytest.mark.asyncio
    async def test_check_timeout(self, checker):
        """Test health check timeout."""
        import asyncio

        async def slow_check():
            await asyncio.sleep(10)  # Will timeout
            return HealthStatus.HEALTHY, "OK", {}

        checker.register("slow", slow_check)
        result = await checker.check_component("slow")

        assert result.status == HealthStatus.UNHEALTHY
        assert "timed out" in result.message.lower()

    @pytest.mark.asyncio
    async def test_check_exception(self, checker):
        """Test health check that raises exception."""
        async def failing_check():
            raise ValueError("Test error")

        checker.register("failing", failing_check)
        result = await checker.check_component("failing")

        assert result.status == HealthStatus.UNHEALTHY
        assert "Test error" in result.message

    @pytest.mark.asyncio
    async def test_check_all_healthy(self, checker):
        """Test checking all components when healthy."""
        async def healthy():
            return HealthStatus.HEALTHY, "OK", {}

        checker.register("a", healthy)
        checker.register("b", healthy)

        result = await checker.check_all()
        assert result.status == HealthStatus.HEALTHY
        assert len(result.components) == 2

    @pytest.mark.asyncio
    async def test_check_all_degraded(self, checker):
        """Test overall status is degraded when one component is degraded."""
        async def healthy():
            return HealthStatus.HEALTHY, "OK", {}

        async def degraded():
            return HealthStatus.DEGRADED, "Slow", {}

        checker.register("a", healthy)
        checker.register("b", degraded)

        result = await checker.check_all()
        assert result.status == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_check_all_unhealthy(self, checker):
        """Test overall status is unhealthy when any component fails."""
        async def healthy():
            return HealthStatus.HEALTHY, "OK", {}

        async def unhealthy():
            return HealthStatus.UNHEALTHY, "Down", {}

        checker.register("a", healthy)
        checker.register("b", unhealthy)

        result = await checker.check_all()
        assert result.status == HealthStatus.UNHEALTHY


class TestDefaultHealthChecks:
    """Tests for default health check functions."""

    @pytest.mark.asyncio
    async def test_check_memory(self):
        """Test memory health check."""
        status, message, details = await check_memory()

        # Should return a valid status
        assert status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]
        assert message is not None

    @pytest.mark.asyncio
    async def test_check_disk(self):
        """Test disk health check."""
        status, message, details = await check_disk()

        # Should return a valid status
        assert status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]
        assert message is not None
