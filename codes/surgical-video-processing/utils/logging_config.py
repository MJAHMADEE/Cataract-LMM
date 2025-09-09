"""
Comprehensive Logging Configuration

This module provides professional logging configuration with multiple
handlers, formatters, and logging levels specifically designed for
surgical video processing operations.
"""

import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for better readability"""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[94m",  # Blue
        "INFO": "\033[92m",  # Green
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",  # Red
        "CRITICAL": "\033[95m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        reset_color = self.COLORS["RESET"]

        # Apply color to the entire log message
        record.levelname = f"{log_color}{record.levelname}{reset_color}"
        record.msg = f"{log_color}{record.msg}{reset_color}"

        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields if present
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "getMessage",
                "exc_info",
                "exc_text",
                "stack_info",
            ]:
                log_entry[key] = value

        return json.dumps(log_entry)


class ProcessingLogFilter(logging.Filter):
    """Custom filter for processing-related logs"""

    def __init__(
        self,
        include_modules: Optional[list] = None,
        exclude_modules: Optional[list] = None,
    ):
        super().__init__()
        self.include_modules = include_modules or []
        self.exclude_modules = exclude_modules or []

    def filter(self, record):
        # If include_modules is specified, only include those modules
        if self.include_modules:
            return any(module in record.name for module in self.include_modules)

        # If exclude_modules is specified, exclude those modules
        if self.exclude_modules:
            return not any(module in record.name for module in self.exclude_modules)

        return True


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console_output: bool = True,
    json_output: bool = False,
    max_file_size: int = 50 * 1024 * 1024,  # 50MB
    backup_count: int = 5,
    include_modules: Optional[list] = None,
    exclude_modules: Optional[list] = None,
) -> logging.Logger:
    """
    Set up comprehensive logging configuration

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        console_output: Whether to output to console
        json_output: Whether to use JSON formatting for file output
        max_file_size: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        include_modules: List of modules to include in logging
        exclude_modules: List of modules to exclude from logging

    Returns:
        Configured logger instance
    """

    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Clear any existing handlers
    logger.handlers.clear()

    # Create custom filter
    log_filter = ProcessingLogFilter(include_modules, exclude_modules)

    # Console handler with colored output
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        if sys.stdout.isatty():  # Only use colors if output is a terminal
            console_formatter = ColoredFormatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        else:
            console_formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

        console_handler.setFormatter(console_formatter)
        console_handler.addFilter(log_filter)
        logger.addHandler(console_handler)

    # File handler with rotation
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_file_size, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setLevel(level)

        if json_output:
            file_formatter = JSONFormatter()
        else:
            file_formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s [%(filename)s:%(lineno)d] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

        file_handler.setFormatter(file_formatter)
        file_handler.addFilter(log_filter)
        logger.addHandler(file_handler)

    # Error file handler (separate file for errors and above)
    if log_file:
        error_log_file = log_file.replace(".log", "_errors.log")
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)

        error_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s [%(filename)s:%(lineno)d] %(message)s\n"
            "Function: %(funcName)s\n"
            "Module: %(module)s\n"
            "%(message)s\n" + "-" * 80,
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        error_handler.setFormatter(error_formatter)
        error_handler.addFilter(log_filter)
        logger.addHandler(error_handler)

    return logger


def get_processing_logger(name: str) -> logging.Logger:
    """
    Get a logger configured for processing operations

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger
    """
    return logging.getLogger(f"surgical_video_processing.{name}")


def log_processing_start(
    logger: logging.Logger, operation: str, input_path: str, **kwargs
):
    """Log the start of a processing operation"""
    logger.info(f"Starting {operation}")
    logger.info(f"Input: {input_path}")
    for key, value in kwargs.items():
        logger.info(f"{key.replace('_', ' ').title()}: {value}")


def log_processing_end(
    logger: logging.Logger,
    operation: str,
    success: bool,
    duration: float,
    output_path: Optional[str] = None,
    **kwargs,
):
    """Log the end of a processing operation"""
    status = "COMPLETED" if success else "FAILED"
    logger.info(f"{operation} {status} in {duration:.2f} seconds")

    if output_path:
        logger.info(f"Output: {output_path}")

    for key, value in kwargs.items():
        logger.info(f"{key.replace('_', ' ').title()}: {value}")


def log_error_with_context(
    logger: logging.Logger, error: Exception, context: Dict[str, Any]
):
    """Log an error with additional context information"""
    logger.error(f"Error: {str(error)}")
    logger.error(f"Error Type: {type(error).__name__}")

    for key, value in context.items():
        logger.error(f"Context - {key}: {value}")

    # Log the full traceback at debug level
    logger.debug("Full traceback:", exc_info=True)


def create_performance_logger(log_file: str) -> logging.Logger:
    """
    Create a specialized logger for performance metrics

    Args:
        log_file: Path to performance log file

    Returns:
        Performance logger
    """
    perf_logger = logging.getLogger("performance")
    perf_logger.setLevel(logging.INFO)

    # Remove existing handlers
    perf_logger.handlers.clear()

    # Create file handler for performance logs
    handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=20 * 1024 * 1024, backupCount=10  # 20MB
    )

    # Use JSON format for performance data
    formatter = JSONFormatter()
    handler.setFormatter(formatter)

    perf_logger.addHandler(handler)
    perf_logger.propagate = False  # Don't propagate to root logger

    return perf_logger


def log_performance_metrics(
    logger: logging.Logger, operation: str, metrics: Dict[str, Any]
):
    """
    Log performance metrics in a structured format

    Args:
        logger: Performance logger
        operation: Name of the operation
        metrics: Dictionary of performance metrics
    """
    log_data = {
        "operation": operation,
        "timestamp": datetime.now().isoformat(),
        **metrics,
    }

    # Use extra parameter to pass structured data
    logger.info("Performance metrics", extra=log_data)


class LoggingContext:
    """Context manager for enhanced logging with automatic cleanup"""

    def __init__(
        self, logger: logging.Logger, operation: str, input_path: str, **context
    ):
        self.logger = logger
        self.operation = operation
        self.input_path = input_path
        self.context = context
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        log_processing_start(
            self.logger, self.operation, self.input_path, **self.context
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        success = exc_type is None

        if success:
            log_processing_end(
                self.logger, self.operation, True, duration, **self.context
            )
        else:
            log_error_with_context(
                self.logger,
                exc_val,
                {
                    "operation": self.operation,
                    "input_path": self.input_path,
                    "duration": duration,
                    **self.context,
                },
            )

        return False  # Don't suppress exceptions


# Pre-configured loggers for common use cases
def get_main_logger() -> logging.Logger:
    """Get the main application logger"""
    return logging.getLogger("surgical_video_processing.main")


def get_core_logger() -> logging.Logger:
    """Get the core processing logger"""
    return logging.getLogger("surgical_video_processing.core")


def get_pipeline_logger() -> logging.Logger:
    """Get the pipeline logger"""
    return logging.getLogger("surgical_video_processing.pipelines")


def get_quality_logger() -> logging.Logger:
    """Get the quality control logger"""
    return logging.getLogger("surgical_video_processing.quality_control")


def get_compression_logger() -> logging.Logger:
    """Get the compression logger"""
    return logging.getLogger("surgical_video_processing.compression")


def get_metadata_logger() -> logging.Logger:
    """Get the metadata logger"""
    return logging.getLogger("surgical_video_processing.metadata")
