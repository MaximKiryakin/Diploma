import logging
import sys
import os
from datetime import datetime
import pandas as pd

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
        _configured_loggers (set): Track which loggers have been configured
    """

    _log_file: str = f"logs/app_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    _initialized: bool = False
    _configured_loggers: set = set()  # Class-level tracking of configured loggers

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
        
        # IMPORTANT: Set propagate to False to prevent duplicate output
        # Each logger manages its own handlers independently
        self.logger.propagate = False
        
        # Configure handlers for this logger instance ONLY if not already configured
        self._setup_handlers(format)

    def _setup_handlers(self, format: str) -> None:
        """
        Configure logging handlers for this logger instance.

        Sets up both console and file handlers with consistent formatting.
        Creates log directory if it doesn't exist.
        Uses class-level tracking to ensure handlers are only added once per logger name.

        Args:
            format: Format string for log messages
        """
        # Check if THIS SPECIFIC logger name has already been configured
        logger_name = self.logger.name
        if logger_name in Logger._configured_loggers:
            return
            
        formatter = logging.Formatter(format)

        log_dir = os.path.dirname(Logger._log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        # Add console handler - using stderr to avoid conflicts with Jupyter output
        # sys.stderr works better than sys.stdout in Jupyter notebooks  
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Add file handler
        file_handler = logging.FileHandler(Logger._log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Mark this logger as configured
        Logger._configured_loggers.add(logger_name)

    def get_logger(self) -> logging.Logger:
        """
        Get the configured logger instance.

        Returns:
            Configured logging.Logger instance
        """
        return self.logger

    def log_missing_values_summary(
        self, missing_dict: dict, title: str = "Missing Values Summary"
    ) -> None:
        """
        Log a pretty formatted summary of missing values as a DataFrame.

        Args:
            missing_dict: Dictionary with column names as keys and missing value ratios as values
            title: Title for the summary table
        """
        if not missing_dict:
            self.logger.info(f"{title}: No missing values to report")
            return

        # Create DataFrame for pretty display
        summary_data = []
        for col_name, missing_ratio in missing_dict.items():
            summary_data.append(
                {
                    "Column": col_name,
                    "Missing Count": int(missing_ratio * 100)
                    if missing_ratio > 1
                    else "N/A",
                    "Missing %": f"{missing_ratio * 100:.2f}%",
                }
            )

        df_summary = pd.DataFrame(summary_data)

        # Log the title
        self.logger.info("=" * 60)
        self.logger.info(title)
        self.logger.info("=" * 60)

        # Log each row of the DataFrame
        for _, row in df_summary.iterrows():
            self.logger.info(
                f"  • {row['Column']:<25} Missing: {row['Missing %']:>8}"
            )

        self.logger.info("=" * 60)
