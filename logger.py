"""Logging configuration for diraqBench"""
import logging
import sys
from pathlib import Path
from typing import Optional


class DiraqBenchLogger:
    """Custom logger for diraqBench with consistent formatting"""
    
    def __init__(self, name: str = "diraqbench"):
        self.name = name
        self.logger = logging.getLogger(name)
        self._configured = False
    
    def configure(self, 
                 level: str = "INFO",
                 log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                 log_file: Optional[str] = None,
                 console: bool = True):
        """Configure the logger with specified settings"""
        
        if self._configured:
            return
        
        # Set level
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        self.logger.setLevel(numeric_level)
        
        # Create formatter
        formatter = logging.Formatter(log_format)
        
        # Console handler
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(numeric_level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        self._configured = True
    
    def get_logger(self, module_name: Optional[str] = None) -> logging.Logger:
        """Get a logger instance for a specific module"""
        if module_name:
            return logging.getLogger(f"{self.name}.{module_name}")
        return self.logger


# Global logger instance
_logger_instance = None


def setup_logging(level: str = "INFO",
                 log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                 log_file: Optional[str] = "diraqbench.log",
                 console: bool = True):
    """Setup global logging configuration"""
    global _logger_instance
    
    if _logger_instance is None:
        _logger_instance = DiraqBenchLogger()
    
    _logger_instance.configure(level, log_format, log_file, console)
    return _logger_instance


def get_logger(module_name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance for a module"""
    global _logger_instance
    
    if _logger_instance is None:
        # Use default configuration if not set up
        _logger_instance = setup_logging()
    
    return _logger_instance.get_logger(module_name)


class LoggerMixin:
    """Mixin class to add logging capability to any class"""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class"""
        module_name = self.__class__.__module__
        class_name = self.__class__.__name__
        return get_logger(f"{module_name}.{class_name}")


# Convenience functions for backward compatibility
def log_info(message: str, module: Optional[str] = None):
    """Log an info message"""
    get_logger(module).info(message)


def log_error(message: str, module: Optional[str] = None, exc_info: bool = False):
    """Log an error message"""
    get_logger(module).error(message, exc_info=exc_info)


def log_warning(message: str, module: Optional[str] = None):
    """Log a warning message"""
    get_logger(module).warning(message)


def log_debug(message: str, module: Optional[str] = None):
    """Log a debug message"""
    get_logger(module).debug(message)