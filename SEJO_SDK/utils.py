"""
Utility class providing common helper methods for the SEJO SDK.
"""

import os
import json
import logging
from typing import Any, Dict, Optional, Union
from pathlib import Path


class Utils:
    """
    A utility class providing common helper methods for the SEJO SDK.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the Utils class.
        
        Args:
            logger: Optional logger instance. If not provided, a default logger will be created.
        """
        self.logger = logger or self._setup_default_logger()
    
    @staticmethod
    def _setup_default_logger() -> logging.Logger:
        """Set up a default logger if none is provided."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    @staticmethod
    def load_json_file(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load and parse a JSON file.
        
        Args:
            file_path: Path to the JSON file.
            
        Returns:
            Parsed JSON data as a dictionary.
            
        Raises:
            FileNotFoundError: If the file doesn't exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def save_to_json(data: Any, file_path: Union[str, Path], indent: int = 4) -> None:
        """
        Save data to a JSON file.
        
        Args:
            data: Data to be saved as JSON.
            file_path: Path where to save the JSON file.
            indent: Indentation level for pretty-printing.
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
    
    @staticmethod
    def get_env_variable(key: str, default: Optional[str] = None) -> str:
        """
        Get an environment variable or return a default value.
        
        Args:
            key: Environment variable name.
            default: Default value if the variable is not set.
            
        Returns:
            The environment variable value or default.
            
        Raises:
            ValueError: If the environment variable is not set and no default is provided.
        """
        value = os.getenv(key, default)
        if value is None:
            raise ValueError(f"Environment variable {key} is not set and no default provided.")
        return value
    
    def log_info(self, message: str) -> None:
        """Log an info message."""
        self.logger.info(message)
    
    def log_error(self, message: str, exc_info: bool = False) -> None:
        """
        Log an error message.
        
        Args:
            message: Error message to log.
            exc_info: Whether to include exception info if available.
        """
        self.logger.error(message, exc_info=exc_info)
    
    def log_warning(self, message: str) -> None:
        """Log a warning message."""
        self.logger.warning(message)
    
    def log_debug(self, message: str) -> None:
        """Log a debug message."""
        self.logger.debug(message)