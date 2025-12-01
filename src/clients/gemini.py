"""Google Gemini client for text generation."""

import logging
from typing import AsyncIterator

import google.generativeai as genai

from src.config import Settings

logger = logging.getLogger(__name__)


class GeminiClient:
    """Google Gemini client for text generation."""

    def __init__(self, settings: Settings):
        """Initialize Gemini client.

        Args:
            settings: Application settings with Gemini configuration
        """
        self.settings = settings
        genai.configure(api_key=settings.gemini_api_key)
        self._model = genai.GenerativeModel(settings.gemini_model)

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
            system: System prompt (prepended to user prompt for Gemini)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            model: Model to use (defaults to settings)

        Returns:
            Generated text
        """
        try:
            # Create model with specific config if needed
            if model and model != self.settings.gemini_model:
                gen_model = genai.GenerativeModel(model)
            else:
                gen_model = self._model

            # Combine system prompt with user prompt
            full_prompt = f"{system}\n\n{prompt}" if system else prompt

            # Generate content
            response = await gen_model.generate_content_async(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                ),
            )

            result = response.text
            logger.info(f"Generated {len(result)} characters")
            return result

        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
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
        try:
            # Create model with specific config if needed
            if model and model != self.settings.gemini_model:
                gen_model = genai.GenerativeModel(model)
            else:
                gen_model = self._model

            # Combine system prompt with user prompt
            full_prompt = f"{system}\n\n{prompt}" if system else prompt

            # Stream content
            response = await gen_model.generate_content_async(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                ),
                stream=True,
            )

            async for chunk in response:
                if chunk.text:
                    yield chunk.text

            logger.info("Completed streaming generation")

        except Exception as e:
            logger.error(f"Gemini streaming failed: {e}")
            raise

    async def close(self):
        """Close Gemini client (no-op for Gemini SDK)."""
        logger.info("Gemini client closed")
