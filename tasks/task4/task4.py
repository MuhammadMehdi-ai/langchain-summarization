"""
Task 4: Agent with summarization tool (modular + LLM analysis).
"""

from app.services.agent_service import AgentService
from app.services.summarizer_service import SentenceSummarizer
from app.services.tool_factory import ToolFactory
from app.core.llm_factory import LLMFactory
from app.services.analysis_prompts import AnalysisPrompts
from app.utils.logger import Logger

logger = Logger.get_logger(__name__)


def main():
    """
    Execute Task 4:
    - Create summarization tool
    - Run agent on clear and vague inputs
    - Analyze behavior using LLM
    """
    try:
        # ---------------- INITIALIZATION ----------------
        logger.info("Initializing services")

        summarizer = SentenceSummarizer(3)
        llm = LLMFactory.create_llm()

        summarizer_tool = ToolFactory.create_summarizer_tool(summarizer)
        agent = AgentService([summarizer_tool])

        # ---------------- TEST 1: CLEAR INPUT ----------------
        logger.info("Running Test 1: Clear summarization request")

        text = """
        Artificial intelligence is transforming healthcare by enabling faster and more accurate diagnoses,
        personalized treatment plans, and improved patient outcomes. AI algorithms analyze large datasets
        to detect patterns and support clinical decisions. It is widely used in medical imaging, drug discovery,
        and patient monitoring systems. However, challenges such as data privacy, bias, and ethical concerns remain.
        """

        response_1 = agent.run(
            f"Summarize the impact of AI on healthcare:\n{text}"
        )

        print("\n--- Test 1 Output ---\n")
        print(response_1)

        # ---------------- TEST 2: VAGUE INPUT ----------------
        logger.info("Running Test 2: Vague query")

        response_2 = agent.run("Summarize something interesting")

        print("\n--- Test 2 Output ---\n")
        print(response_2)

        # ---------------- LLM ANALYSIS ----------------
        logger.info("Generating LLM-based analysis")

        analysis_prompt = AnalysisPrompts.agent_behavior_analysis()
        analysis_chain = analysis_prompt | llm

        analysis_result = analysis_chain.invoke({
            "test1": response_1,
            "test2": response_2
        })

        print("\n--- LLM Analysis ---\n")
        print(analysis_result.content)

        logger.info("Task 4 completed successfully")

    except Exception as error:
        logger.error(f"Task 4 failed: {error}")
        raise


if __name__ == "__main__":
    main()