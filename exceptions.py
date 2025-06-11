"""
Custom exception classes for the question processing system.
"""


class QuestionProcessingError(Exception):
    """Base exception class for question processing errors"""
    pass


class APIKeyError(QuestionProcessingError):
    """Raised when API key is missing or invalid"""
    pass


class APIResponseError(QuestionProcessingError):
    """Raised when API call fails or returns unexpected response"""
    pass


class APIRateLimitError(QuestionProcessingError):
    """Raised when API rate limit is exceeded"""
    pass


class ModelNotAvailableError(QuestionProcessingError):
    """Raised when requested model or provider is not supported"""
    pass


class ImageProcessingError(QuestionProcessingError):
    """Raised when image encoding or processing fails"""
    pass


class JSONProcessingError(QuestionProcessingError):
    """Raised when JSON parsing or processing fails"""
    pass


class ValidationError(QuestionProcessingError):
    """Raised when input validation fails"""
    pass


class PathNotFoundError(QuestionProcessingError):
    """Raised when required file or directory is not found"""
    pass


class ConfigurationError(QuestionProcessingError):
    """Raised when configuration is invalid or missing"""
    pass


class ProcessingTimeoutError(QuestionProcessingError):
    """Raised when processing takes too long"""
    pass 