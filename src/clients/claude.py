"""Anthropic Claude client for text generation."""

import logging
from typing import AsyncIterator
from anthropic import AsyncAnthropic

from src.config import Settings

logger = logging.getLogger(__name__)


class ClaudeClient:
    """Anthropic Claude client for text generation."""

    def __init__(self, settings: Settings):
        """Initialize Claude client.

        Args:
            settings: Application settings with Anthropic configuration
        """
        self.settings = settings
        self._client = AsyncAnthropic(api_key=settings.anthropic_api_key)

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        model: str | None = None,
    ) -> str:
        """Generate text from prompt.

        Args:
            prompt: User prompt
            system: System prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            model: Model to use (defaults to settings)

        Returns:
            Generated text
        """
        model = model or self.settings.claude_model

        try:
            message = await self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system or "",
                messages=[{"role": "user", "content": prompt}],
            )
            result = message.content[0].text
            logger.info(f"Generated {len(result)} characters")
            return result
        except Exception as e:
            logger.error(f"Claude generation failed: {e}")
            raise

    async def generate_stream(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        model: str | None = None,
    ) -> AsyncIterator[str]:
        """Generate text from prompt with streaming.

        Args:
            prompt: User prompt
            system: System prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            model: Model to use (defaults to settings)

        Yields:
            Text chunks as they are generated
        """
        model = model or self.settings.claude_model

        try:
            async with self._client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system or "",
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                async for text in stream.text_stream:
                    yield text
            logger.info("Completed streaming generation")
        except Exception as e:
            logger.error(f"Claude streaming failed: {e}")
            raise

    async def close(self):
        """Close Claude client."""
        await self._client.close()
        logger.info("Claude client closed")
