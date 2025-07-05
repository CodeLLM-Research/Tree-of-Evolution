"""
Custom exceptions for LLM operations.
"""


class APIError(Exception):
    """Exception raised when API calls fail."""

    def __init__(self, message: str, status_code: int = None):
        super().__init__(message)
        self.status_code = status_code
