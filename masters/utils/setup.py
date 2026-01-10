import warnings
import logging


def setup_enviroment() -> None:
    """Sets up the environment by silencing loggers and ignoring specific warnings."""
    # Completely silence cmdstanpy and fbprophet
    logging.getLogger("cmdstanpy").addHandler(logging.NullHandler())
    logging.getLogger("cmdstanpy").propagate = False
    logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)

    warnings.filterwarnings("ignore", category=UserWarning, message="Unable to import Axes3D")
    warnings.filterwarnings("ignore", category=UserWarning, message="Workbook contains no default style")
    warnings.filterwarnings(
        "ignore", category=UserWarning, message="YF.download() has changed argument auto_adjust default to True"
    )
    warnings.filterwarnings("ignore", message="IProgress not found")
