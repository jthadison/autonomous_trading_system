import logging
import structlog
import os
from pathlib import Path
#from .settings import settings

def setup_logging():
    """Configure structured logging for the application (Windows compatible)"""
    
    # Create logs directory (Windows compatible)
    os.makedirs("logs", exist_ok=True)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if os.getenv("LOG_FORMAT") == "json" else structlog.dev.ConsoleRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging (Windows file handling)
    log_file = "logs/trading_system.log"
    logging.basicConfig(
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ],
        level="INFO"#os.getenv("LOG_LEVEL").upper()
    )

# Initialize logger
setup_logging()
logger = structlog.get_logger()