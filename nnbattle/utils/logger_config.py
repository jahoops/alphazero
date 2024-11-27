import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def setup_logger(level=logging.INFO):
    """Configure the root logger with consistent settings."""
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear any existing handlers
    root_logger.handlers = []

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Optional: Add file handler
    file_handler = logging.FileHandler('alphazero_training.log')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    return root_logger

# Create a global logger instance
logger = setup_logger()

def set_log_level(level=logging.WARNING):
    """Set the global log level for all loggers."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.getLogger().setLevel(level)

logger = logging.getLogger(__name__)
set_log_level(logging.WARNING)  # Default log level
