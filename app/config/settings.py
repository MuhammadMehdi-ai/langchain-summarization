"""
Environment configuration loader.
"""

import os
from dotenv import load_dotenv


class Settings:
    """Loads environment variables and exposes them safely."""

    def __init__(self):
        load_dotenv()

        self.azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = os.getenv("ENDPOINT_URL")
        self.deployment = os.getenv("DEPLOYMENT_NAME")
        self.api_version = os.getenv("API_VERSION")
        self.embedding_deployment = os.getenv("EMBEDDING_DEPLOYMENT_NAME")

        self._validate()

    def _validate(self):
        """Validate required environment variables."""
        if not self.azure_api_key:
            raise ValueError("Missing AZURE_OPENAI_API_KEY")