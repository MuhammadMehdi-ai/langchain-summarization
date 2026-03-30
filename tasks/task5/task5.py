"""
Task 5: Combining Chains, Retrievers, and Agents (with LLM analysis).
"""

from app.services.agent_service import AgentService
from app.services.retriever_service import RetrieverService
from app.services.summarizer_service import SentenceSummarizer
from app.services.tool_factory import ToolFactory
from app.core.llm_factory import LLMFactory
from app.services.analysis_prompts import AnalysisPrompts
from app.utils.logger import Logger

logger = Logger.get_logger(__name__)


def word_count(text: str) -> str:
    """Count words in text."""
    return f"Word count: {len(text.split())}"


def main():
    """Run agent pipeline with analysis."""
    try:
        # ---------------- INIT ----------------
        retriever = RetrieverService("ai_intro.txt")
        summarizer = SentenceSummarizer(3)
        llm = LLMFactory.create_llm()

        retriever_tool = ToolFactory.create_tool(
            name="TextRetriever",
            func=retriever.retrieve,
            description="Retrieve relevant text from the AI document based on a query."
        )

        summarizer_tool = ToolFactory.create_tool(
            name="TextSummarizer",
            func=summarizer.summarize,
            description="Summarize provided text into exactly 3 sentences."
        )

        wordcount_tool = ToolFactory.create_tool(
            name="WordCounter",
            func=word_count,
            description="Count the number of words in the summary."
        )

        agent = AgentService([retriever_tool, summarizer_tool, wordcount_tool])

        # ---------------- TEST 1 ----------------
        logger.info("Running Test 1")

        query1 = "Find and summarize text about AI breakthroughs from the document."

        response1 = agent.run(query1)

        print("\n--- Test 1 Output ---\n")
        print(response1)

        # ---------------- TEST 2 ----------------
        logger.info("Running Test 2")

        query2 = """
        Step 1: Retrieve relevant text about AI breakthroughs from the document.
        Step 2: Summarize the retrieved text in exactly 3 sentences.
        Step 3: Count the number of words in the summary.
        Return both summary and word count.
        """

        response2 = agent.run(query2)

        print("\n--- Test 2 Output ---\n")
        print(response2)

        # ---------------- LLM ANALYSIS ----------------
        logger.info("Generating pipeline analysis")

        analysis_prompt = AnalysisPrompts.pipeline_analysis()
        analysis_chain = analysis_prompt | llm

        analysis_result = analysis_chain.invoke({
            "test1": response1,
            "test2": response2
        })

        print("\n--- LLM Analysis ---\n")
        print(analysis_result.content)

        logger.info("Task 5 completed successfully")

    except Exception as error:
        logger.error(f"Task 5 failed: {error}")
        raise


if __name__ == "__main__":
    main()