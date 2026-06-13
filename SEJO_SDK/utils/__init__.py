"""Common helper methods for the SEJO SDK."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional, Union

from SEJO_SDK.utils.postgresql_connector import PostgresqlConnector


class Utils:
    """Utility class providing common helper methods for the SEJO SDK."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or self._setup_default_logger()

    @staticmethod
    def _setup_default_logger() -> logging.Logger:
        """Set up a default logger if none is provided."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    @staticmethod
    def load_json_file(file_path: Union[str, Path]) -> dict[str, Any]:
        """Load and parse a JSON file."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def save_to_json(data: Any, file_path: Union[str, Path], indent: int = 4) -> None:
        """Save data to a JSON file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=indent, ensure_ascii=False)

    @staticmethod
    def get_env_variable(key: str, default: Optional[str] = None) -> str:
        """Get an environment variable or return a default value."""
        value = os.getenv(key, default)
        if value is None:
            raise ValueError(
                f"Environment variable {key} is not set and no default provided."
            )
        return value

    def log_info(self, message: str) -> None:
        """Log an info message."""
        self.logger.info(message)

    def log_error(self, message: str, exc_info: bool = False) -> None:
        """Log an error message."""
        self.logger.error(message, exc_info=exc_info)

    def log_warning(self, message: str) -> None:
        """Log a warning message."""
        self.logger.warning(message)

    def log_debug(self, message: str) -> None:
        """Log a debug message."""
        self.logger.debug(message)


__all__ = ["PostgresqlConnector", "Utils"]
