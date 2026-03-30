"""
Task 11: Agent with external tools (complete implementation).
"""

from datetime import datetime
from app.services.agent_service import AgentService
from app.services.retriever_service import RetrieverService
from app.services.summarizer_service import SentenceSummarizer
from app.services.tool_factory import ToolFactory
from app.utils.logger import Logger

logger = Logger.get_logger(__name__)


def get_current_date(_: str) -> str:
    """Return today's date."""
    return datetime.now().strftime("%Y-%m-%d")


def mock_web_search(query: str) -> str:
    """Return mock 50-word AI trends."""
    return (
        "Recent developments in artificial intelligence highlight rapid progress in generative AI, "
        "large language models, and automation technologies. Organizations are adopting AI to improve "
        "efficiency, decision-making, and innovation. However, concerns about ethics, bias, data privacy, "
        "and workforce impact continue to shape discussions around responsible AI deployment."
    )


def main():
    """Run agent with extended tools."""
    try:
        # ---------------- SERVICES ----------------
        retriever = RetrieverService("ai_intro.txt")
        summarizer = SentenceSummarizer(3)

        # ---------------- TOOLS ----------------
        retriever_tool = ToolFactory.create_tool(
            name="TextRetriever",
            func=retriever.retrieve,
            description="Retrieve relevant AI text from the document based on a query."
        )

        summarizer_tool = ToolFactory.create_tool(
            name="TextSummarizer",
            func=summarizer.summarize,
            description="Summarize provided text into exactly 3 sentences."
        )

        wordcount_tool = ToolFactory.create_tool(
            name="WordCounter",
            func=lambda text: f"Word count: {len(text.split())}",
            description="Count number of words in given text."
        )

        date_tool = ToolFactory.create_tool(
            name="CurrentDate",
            func=get_current_date,
            description="Return today's date in YYYY-MM-DD format."
        )

        search_tool = ToolFactory.create_tool(
            name="WebSearch",
            func=mock_web_search,
            description="Search for recent AI trends and return relevant information."
        )

        # ---------------- AGENT ----------------
        agent = AgentService([
            retriever_tool,
            summarizer_tool,
            wordcount_tool,
            date_tool,
            search_tool
        ])

        # ---------------- TEST 1 ----------------
        logger.info("Running Test 1: Summarize + Date")

        text_input = """
Artificial intelligence is transforming industries by improving efficiency and automation. 
It is widely used in healthcare for diagnosis, finance for fraud detection, and transportation 
for self-driving systems. AI enhances customer experience through chatbots and recommendation 
systems but also raises concerns about privacy, bias, and job displacement.AI continues to evolve rapidly, with ongoing research in natural language processing, computer vision,
and reinforcement learning driving innovation across various sectors. Organizations are increasingly adopting AI to gain competitive advantages, optimize operations, and create new products and services. However, ethical considerations and regulatory frameworks are crucial to ensure responsible AI development and deployment.
Also, the integration of AI with other emerging technologies like IoT and blockchain is opening up new possibilities for data analysis, security, and automation. As AI technology advances, it is essential to address challenges related to transparency, accountability, and societal impact to maximize its benefits while minimizing risks.
"""

        query1 = f"""
Summarize this text in 3 sentences and tell me today's date:

{text_input}
"""

        response1 = agent.run(query1)

        print("\n--- Test 1 Output ---\n")
        print(response1)

        # ---------------- TEST 2 ----------------
        logger.info("Running Test 2: Summarize + Search")

        query2 = """
Summarize AI trends and search for recent updates.
Return both the summary and search results.
"""

        response2 = agent.run(query2)

        print("\n--- Test 2 Output ---\n")
        print(response2)

        logger.info("Task 11 completed successfully")

    except Exception as error:
        logger.error(f"Task 11 failed: {error}")
        raise


if __name__ == "__main__":
    main()