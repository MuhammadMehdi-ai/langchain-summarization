"""
Factory for embeddings.
"""

from langchain_openai import AzureOpenAIEmbeddings
from app.config.settings import Settings


class EmbeddingsFactory:
    """Factory class for embeddings."""

    @staticmethod
    def create_embeddings():
        """Create embeddings instance."""
        settings = Settings()

        return AzureOpenAIEmbeddings(
            api_key=settings.azure_api_key,
            azure_endpoint=settings.endpoint,
            deployment=settings.embedding_deployment,
            api_version=settings.api_version,
        )