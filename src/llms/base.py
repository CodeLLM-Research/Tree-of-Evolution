"""
Base classes for LLM clients.
"""

from abc import ABC, abstractmethod
from typing import Optional


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize the LLM client.
        
        Args:
            api_key: API key for authentication
            **kwargs: Additional configuration parameters
        """
        self.api_key = api_key
        self.config = kwargs

    @abstractmethod
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
        Send a request to the LLM and return the response.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds
            **kwargs: Additional model-specific parameters
            
        Returns:
            The response text or None if failed
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the LLM service is available.
        
        Returns:
            True if available, False otherwise
        """
        raise NotImplementedError("Subclasses must implement this method")
