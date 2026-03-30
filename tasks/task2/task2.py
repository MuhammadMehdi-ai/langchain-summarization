"""
Task 2: Summarization with logging and exception handling.
"""

from app.services.summarizer_service import SentenceSummarizer
from app.utils.logger import Logger

logger = Logger.get_logger(__name__)


def main():
    """Run summarization examples."""
    try:
        text = """Artificial intelligence (AI) refers to the simulation of human 
        intelligence in machines that are programmed to think and learn.
        AI has various applications, including natural language processing,
        computer vision, and robotics.
        The goal of AI is to create systems that can perform tasks that 
        typically require human intelligence, such as decision-making,
        problem-solving, and language understanding.As AI continues to evolve,
        it has the potential to transform industries and improve our daily lives in numerous ways."""

        summarizer_3 = SentenceSummarizer(3)
        summarizer_1 = SentenceSummarizer(1)

        result_3 = summarizer_3.summarize(text)
        result_1 = summarizer_1.summarize(text)

        logger.info("Summarization completed")

        print("3 Sentence Summary:\n", result_3)
        print("\n1 Sentence Summary:\n", result_1)

    except Exception as error:
        logger.error(f"Task 2 failed: {error}")
        raise


if __name__ == "__main__":
    main()