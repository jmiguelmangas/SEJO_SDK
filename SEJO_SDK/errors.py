"""Exception hierarchy for SEJO SDK."""


class SejoSDKError(Exception):
    """Base exception for all SDK errors."""


class ProviderDependencyError(SejoSDKError):
    """Raised when an optional provider dependency is not installed."""


class ProviderResponseError(SejoSDKError):
    """Raised when a provider response cannot be parsed."""
