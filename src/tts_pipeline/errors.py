"""Domain-specific exceptions exposed consistently by CLI and HTTP APIs."""


class TTSError(Exception):
    """Base error for expected platform failures."""


class ConfigurationError(TTSError):
    """Raised when resolved configuration is inconsistent."""


class ValidationError(TTSError):
    """Raised when input or dataset content is invalid."""


class CompatibilityError(TTSError):
    """Raised when model, vocabulary, or features are incompatible."""


class OverloadedError(TTSError):
    """Raised when the bounded synthesis queue cannot accept work."""
