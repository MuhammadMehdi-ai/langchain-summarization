"""
Central logging configuration for the application.
"""

import logging


class Logger:
    """Singleton Logger class for application-wide logging."""

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """Create and return a configured logger instance."""
        logger = logging.getLogger(name)

        if not logger.handlers:
            logger.setLevel(logging.INFO)

            console_handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            console_handler.setFormatter(formatter)

            logger.addHandler(console_handler)

        return logger