"""Exception hierarchy for SEJO SDK."""


class SejoSDKError(Exception):
    """Base exception for all SDK errors."""


class ProviderDependencyError(SejoSDKError):
    """Raised when an optional provider dependency is not installed."""


class ProviderResponseError(SejoSDKError):
    """Raised when a provider response cannot be parsed."""


class ToolNotFoundError(SejoSDKError):
    """Raised when an agent is asked to run an unknown tool."""


class ToolExecutionError(SejoSDKError):
    """Raised when a registered tool fails during execution."""
