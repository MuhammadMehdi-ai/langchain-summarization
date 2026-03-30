"""
Predefined reusable tools for the application.
"""

from datetime import datetime
from app.services.tool_factory import ToolFactory


class ToolService:
    """Provides commonly used tools."""

    @staticmethod
    def get_word_count_tool():
        """Word count tool."""

        def count_words(text: str) -> str:
            return f"Word count: {len(text.split())}"

        return ToolFactory.create_tool(
            "WordCounter",
            count_words,
            "Counts number of words in text"
        )

    @staticmethod
    def get_date_tool():
        """Current date tool."""

        def get_date(_: str) -> str:
            return datetime.now().strftime("%Y-%m-%d")

        return ToolFactory.create_tool(
            "CurrentDate",
            get_date,
            "Returns current date"
        )

    @staticmethod
    def get_mock_search_tool():
        """Mock search tool."""

        def search(_: str) -> str:
            return "AI is evolving rapidly with LLMs and automation."

        return ToolFactory.create_tool(
            "WebSearch",
            search,
            "Search AI trends (mock)"
        )