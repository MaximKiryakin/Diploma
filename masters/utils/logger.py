import logging
import sys
import os
from datetime import datetime

logging.getLogger("matplotlib").setLevel(logging.ERROR)


class Logger:
    """
    A wrapper class for standard logging with shared file output.

    This logger ensures all instances write to the same log file and provides
    consistent formatting across the application. It automatically creates
    the log directory if it doesn't exist.

    Attributes:
        _log_file (str): Shared log file path for all instances
        _initialized (bool): Flag tracking initialization status
    """

    _log_file: str = f"logs/app_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    _initialized: bool = False

    def __init__(
        self,
        name: str,
        level: int = logging.INFO,
        format: str = "%(asctime)s:%(name)s:%(levelname)s: %(message)s",
    ) -> None:
        """
        Initialize the logger instance.

        Args:
            name: Logger name (typically __name__)
            level: Logging level (default: INFO)
            format: Message format string
        """
        self.logger: logging.Logger = logging.getLogger(name)
        self.logger.setLevel(level)

        if not Logger._initialized:
            self._setup_handlers(format)
            Logger._initialized = True

    @staticmethod
    def _setup_handlers(format: str) -> None:
        """
        Configure logging handlers.

        Sets up both console and file handlers with consistent formatting.
        Creates log directory if it doesn't exist.

        Args:
            format: Format string for log messages
        """
        formatter = logging.Formatter(format)

        log_dir = os.path.dirname(Logger._log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(Logger._log_file)
        file_handler.setFormatter(formatter)

        root_logger = logging.getLogger()
        for handler in [console_handler, file_handler]:
            root_logger.addHandler(handler)

    def get_logger(self) -> logging.Logger:
        """
        Get the configured logger instance.

        Returns:
            Configured logging.Logger instance
        """
        return self.logger