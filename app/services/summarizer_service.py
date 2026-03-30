"""
Summarization service abstraction.
"""

from abc import ABC, abstractmethod
from langchain_core.prompts import PromptTemplate
from app.core.llm_factory import LLMFactory


class BaseSummarizer(ABC):
    """Abstract base class for summarizers."""

    @abstractmethod
    def summarize(self, text: str) -> str:
        pass


class SentenceSummarizer(BaseSummarizer):
    """Concrete summarizer for N sentence summaries."""

    def __init__(self, num_sentences: int = 3):
        self.llm = LLMFactory.create_llm()
        self.num_sentences = num_sentences

        self.prompt = PromptTemplate.from_template(
            f"Summarize the following text in exactly {num_sentences} sentences:\n{{text}}"
        )

        self.chain = self.prompt | self.llm

    def summarize(self, text: str) -> str:
        """Summarize text."""
        try:
            result = self.chain.invoke({"text": text})
            return result.content
        except Exception as error:
            raise RuntimeError(f"Summarization failed: {error}")