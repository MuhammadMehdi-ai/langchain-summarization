# app/config/env_loader.py
import os
from dotenv import load_dotenv
from app.utils.logger import Logger


class EnvLoader:
    """
    Loads environment variables from .env file.
    """

    def __init__(self):
        self.logger = Logger.get_logger(self.__class__.__name__)

    def load(self):
        """
        Load environment variables from .env file.
        """
        try:
            load_dotenv()
            self.logger.info("Environment variables loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error loading environment variables: {str(e)}")
            raise