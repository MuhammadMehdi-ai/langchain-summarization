"""
Reusable Tool Factory for creating LangChain tools.
"""

from typing import Callable
from langchain.tools import Tool
from app.utils.logger import Logger

logger = Logger.get_logger(__name__)


class ToolFactory:
    """Factory class to create reusable tools."""

    @staticmethod
    def create_tool(name: str, func: Callable, description: str) -> Tool:
        """
        Generic tool creator.

        Args:
            name (str): Tool name
            func (Callable): Function to execute
            description (str): Tool description

        Returns:
            Tool: LangChain Tool instance
        """
        try:
            logger.info(f"Creating tool: {name}")

            return Tool(
                name=name,
                func=func,
                description=description
            )

        except Exception as error:
            logger.error(f"Failed to create tool {name}: {error}")
            raise

    @staticmethod
    def create_summarizer_tool(summarizer) -> Tool:
        """
        Create summarizer tool.

        Args:
            summarizer: Summarizer instance

        Returns:
            Tool
        """
        return ToolFactory.create_tool(
            name="TextSummarizer",
            func=summarizer.summarize,
            description="Summarizes text into exactly 3 sentences"
        )