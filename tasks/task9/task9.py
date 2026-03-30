"""
Task 9: Multi-query vs single-query retrieval comparison.
"""

from langchain.retrievers.multi_query import MultiQueryRetriever
from app.services.retriever_service import RetrieverService
from app.services.summarizer_service import SentenceSummarizer
from app.core.llm_factory import LLMFactory
from app.services.analysis_prompts import AnalysisPrompts
from app.utils.logger import Logger

logger = Logger.get_logger(__name__)


def main():
    """Execute multi-query vs single-query comparison."""
    try:
        retriever_service = RetrieverService("ai_intro.txt")
        llm = LLMFactory.create_llm()
        summarizer = SentenceSummarizer(3)

        query = "AI advancements"

        # ---------------- SINGLE QUERY ----------------
        logger.info("Running single-query retrieval")

        single_retriever = retriever_service.vectorstore.as_retriever()
        single_docs = single_retriever.invoke(query)

        single_text = "\n".join([d.page_content for d in single_docs])
        single_summary = summarizer.summarize(single_text)

        print("\n--- SINGLE QUERY SUMMARY ---\n")
        print(single_summary)

        # ---------------- MULTI QUERY ----------------
        logger.info("Running multi-query retrieval")

        multi_retriever = MultiQueryRetriever.from_llm(
            retriever=single_retriever,
            llm=llm
        )

        multi_docs = multi_retriever.invoke(query)

        multi_text = "\n".join([d.page_content for d in multi_docs])
        multi_summary = summarizer.summarize(multi_text)

        print("\n--- MULTI QUERY SUMMARY ---\n")
        print(multi_summary)

        # ---------------- LLM COMPARISON ----------------
        logger.info("Running LLM-based comparison")

        analysis_prompt = AnalysisPrompts.multi_query_analysis()
        analysis_chain = analysis_prompt | llm

        analysis = analysis_chain.invoke({
            "single": single_summary,
            "multi": multi_summary
        })

        print("\n--- COMPARISON ---\n")
        print(analysis.content)

        logger.info("Task 9 completed successfully")

    except Exception as error:
        logger.error(f"Task 9 failed: {error}")
        raise


if __name__ == "__main__":
    main()