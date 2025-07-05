"""
OpenAI client implementation.
"""

import os
import time
import logging
from typing import Optional, List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

from .base import BaseLLMClient
from .exceptions import APIError

load_dotenv()

logger = logging.getLogger(__name__)


class OpenAIClient(BaseLLMClient):
    """OpenAI API client implementation."""

    DEFAULT_SYSTEM_PROMPT = (
        "You are a professional programming assistant. "
        "You **MUST** carefully follow the user's requests."
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        max_retries: int = 10,
        retry_delay: int = 60,
        **kwargs
    ):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key. If None, will try to get from environment
            model: Model name to use
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            **kwargs: Additional configuration
        """
        super().__init__(api_key, **kwargs)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client = None

        if not self.api_key:
            raise APIError("OPENAI_API_KEY not found in environment variables")

    @property
    def client(self) -> OpenAI:
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def request(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 2048,
        timeout: int = 60,
        **kwargs
    ) -> Optional[str]:
        """
        Send a request to OpenAI API.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds
            **kwargs: Additional parameters
            
        Returns:
            The response text or None if failed
        """
        if system_prompt is None:
            system_prompt = self.DEFAULT_SYSTEM_PROMPT

        messages = self._build_messages(prompt, system_prompt)

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    **kwargs
                )
                return response.choices[0].message.content

            except Exception as e:
                logger.warning("Attempt %d failed: %s", attempt, e)

                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                else:
                    logger.error("Max retries reached. Request failed.")
                    raise APIError(f'API request failed after {self.max_retries} attempts: {e}') from e

        return None

    def _build_messages(self, prompt: str, system_prompt: str) -> List[Dict[str, Any]]:
        """Build messages for OpenAI API."""
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user", 
                "content": [{"type": "text", "text": prompt}],
            },
        ]

    def is_available(self) -> bool:
        """Check if OpenAI API is available."""
        try:
            # Simple test request
            self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
                timeout=10
            )
            return True
        except Exception as e:
            logger.warning("OpenAI API availability check failed: %s", e)
            return False
