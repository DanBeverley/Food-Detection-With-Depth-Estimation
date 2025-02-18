import logging
import sys
from logging.handlers import TimedRotatingFileHandler

def setup_logging(config):
    """Configure hierarchical logging system"""
    logger = logging.getLogger()
    logger.setLevel(config["logging"]["level"])
    # File handler with rotation
    file_handler = TimedRotatingFileHandler(config["logging"]["file"],
                                            when = config["logging"].get("rotation", "midnight"))
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Special logger for data pipeline
    data_logger = logging.getLogger("DataPipeline")
    data_logger.propagate = False
    data_logger.addHandler(logging.FileHandler("log/data_pipeline.log"))
    return logger