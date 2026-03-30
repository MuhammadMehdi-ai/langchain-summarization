"""
Task 1: Environment validation with logging and exception handling.
"""

from app.config.settings import Settings
from app.utils.logger import Logger

logger = Logger.get_logger(__name__)


def main():
    """Load and print environment variables."""
    try:
        settings = Settings()

        logger.info("Environment variables loaded successfully")

        print(settings.azure_api_key)
        print(settings.endpoint)
        print(settings.deployment)
        print(settings.api_version)

    except Exception as error:
        logger.error(f"Task 1 failed: {error}")
        raise


if __name__ == "__main__":
    main()