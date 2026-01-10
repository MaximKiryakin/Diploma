import logging
import os
import types
from typing import Optional
import pandas as pd

# Prevent matplotlib from spamming
logging.getLogger("matplotlib").setLevel(logging.ERROR)

DEFAULT_LOG_FILE = os.path.join("logs", "app.log")

# FORMAT = "%(asctime)s:%(name)s:%(funcName)s:%(levelname)s: %(message)s"
FORMAT = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"


def _log_missing_values_summary(self, dataframe_or_dict, title: str = "Missing Values Summary") -> None:
    """Custom method attached to logger instances. Accepts either a DataFrame or a pre-calculated dict."""
    # If it's a DataFrame, calculate missing values
    if isinstance(dataframe_or_dict, pd.DataFrame):
        missing_dict = (dataframe_or_dict.isna().sum() / len(dataframe_or_dict)).to_dict()
    else:
        missing_dict = dataframe_or_dict

    if not missing_dict:
        self.info(f"{title}: No missing values found")
        return

    data = []
    for col, val in missing_dict.items():
        if val > 0:
            val_str = f"{val:.2%}" if isinstance(val, float) and 0 <= val <= 1 else str(val)
            data.append({"Column": col, "Value": val_str})

    if not data:
        self.info(f"{title}: No missing values found")
        return

    df = pd.DataFrame(data)
    self.log_dataframe(df, title=title)


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
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
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
def log_missing_values_summary(
    logger_obj: logging.Logger, missing_dict: dict, title: str = "Missing Values Summary"
) -> None:
    _log_missing_values_summary(logger_obj, missing_dict, title)


def _log_dataframe(self, df, title: str = None) -> None:
    """Custom method to log a pandas DataFrame nicely."""
    if df is None or df.empty:
        self.info(f"{title}: Empty DataFrame")
        return

    col_widths = []
    for col in df.columns:
        max_data_len = df[col].astype(str).map(len).max() if not df[col].empty else 0
        max_len = max(max_data_len, len(str(col)))
        col_widths.append(max_len)

    header_parts = []
    for i, col in enumerate(df.columns):
        if i == 0:
            header_parts.append(str(col).ljust(col_widths[i]))
        else:
            header_parts.append(str(col).rjust(col_widths[i]))
    header_str = "  ".join(header_parts)

    row_strings = []
    for _, row in df.iterrows():
        row_parts = []
        for i, col in enumerate(df.columns):
            val = str(row[col])
            if i == 0:
                row_parts.append(val.ljust(col_widths[i]))
            else:
                row_parts.append(val.rjust(col_widths[i]))
        row_strings.append("  ".join(row_parts))

    df_str = header_str + "\n" + "\n".join(row_strings)

    if title:
        self.info(f"{title}\n{df_str}")
    else:
        self.info(df_str)


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
