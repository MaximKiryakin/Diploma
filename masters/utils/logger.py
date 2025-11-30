import logging
import sys
import os
import types
from typing import Optional

# Prevent matplotlib from spamming
logging.getLogger("matplotlib").setLevel(logging.ERROR)

DEFAULT_LOG_FILE = os.path.join("logs", "app.log")
FORMAT = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"

def _log_missing_values_summary(self, missing_dict: dict, title: str = "Missing Values Summary") -> None:
    """Custom method attached to logger instances."""
    if not missing_dict:
        self.info(f"{title}: none")
        return

    self.info("=" * 60)
    self.info(f"{title}")
    self.info("-" * 60)
    for col, val in missing_dict.items():
        val_str = f"{val:.2%}" if isinstance(val, float) and 0 <= val <= 1 else str(val)
        self.info(f"{col:<30} | {val_str}")
    self.info("=" * 60)

def Logger(name: str = __name__, level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """
    Factory function that configures and returns a standard logging.Logger.
    Attaches a FileHandler and a StreamHandler (console).
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Clear existing handlers to avoid duplication (notebook friendly)
    if logger.hasHandlers():
        logger.handlers.clear()

    # Double check to ensure no handlers remain
    logger.handlers = []

    formatter = logging.Formatter(FORMAT)

    # 1. File Handler
    log_path = log_file or DEFAULT_LOG_FILE
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 2. Console Handler
    # Use PrintHandler for better compatibility with Jupyter/VS Code
    console_handler = PrintHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Attach custom method dynamically to satisfy existing code usage
    # (logger.log_missing_values_summary)
    logger.log_missing_values_summary = types.MethodType(_log_missing_values_summary, logger)
    logger.log_dataframe = types.MethodType(_log_dataframe, logger)

    return logger

def get_logger(name: str = __name__, level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    return Logger(name, level, log_file)

# Keep standalone function for compatibility if imported directly
def log_missing_values_summary(logger_obj: logging.Logger, missing_dict: dict, title: str = "Missing Values Summary") -> None:
    _log_missing_values_summary(logger_obj, missing_dict, title)



def _log_dataframe(self, df, title: str = None) -> None:
    """Custom method to log a pandas DataFrame nicely."""
    if df is None or df.empty:
        self.info(f"{title}: Empty DataFrame")
        return

    if title:
        self.info("=" * 60)
        self.info(f"{title}")
        self.info("-" * 60)

    # Convert DataFrame to string with borders
    df_str = df.to_string(index=False)
    for line in df_str.split('\n'):
        self.info(line)

    if title:
        self.info("=" * 60)

class PrintHandler(logging.Handler):
    """
    Custom handler that uses print() to output logs.
    This often works better in Jupyter notebooks than writing directly to sys.stdout/stderr.
    """
    def emit(self, record):
        try:
            msg = self.format(record)
            print(msg)
        except Exception:
            self.handleError(record)

