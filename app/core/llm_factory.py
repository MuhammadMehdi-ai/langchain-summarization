"""
Factory for creating LLM instances.
"""

from langchain_openai import AzureChatOpenAI
from app.config.settings import Settings


class LLMFactory:
    """Factory class for AzureChatOpenAI."""

    @staticmethod
    def create_llm(temperature: float = 0):
        """Create LLM instance."""
        settings = Settings()

        return AzureChatOpenAI(
            api_key=settings.azure_api_key,
            azure_endpoint=settings.endpoint,
            deployment_name=settings.deployment,
            api_version=settings.api_version,
            temperature=temperature
        )