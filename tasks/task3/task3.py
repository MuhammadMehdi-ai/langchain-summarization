"""
Task 3: Retriever + Summarizer with logging.
"""

from app.services.retriever_service import RetrieverService
from app.services.summarizer_service import SentenceSummarizer
from app.utils.logger import Logger

logger = Logger.get_logger(__name__)


def main():
    """Execute retrieval and summarization."""
    try:
        retriever = RetrieverService("ai_intro.txt")
        summarizer = SentenceSummarizer(3)

        query = "AI milestones"
        retrieved_text = retriever.retrieve(query)

        logger.info("Text retrieved successfully")

        summary = summarizer.summarize(retrieved_text)

        print("\nRetrieved Text:\n", retrieved_text)
        print("\nSummary:\n", summary)

    except Exception as error:
        logger.error(f"Task 3 failed: {error}")
        raise


if __name__ == "__main__":
    main()