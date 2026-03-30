"""
Task 6: Memory-based summarization (LLM analysis + modular prompts).
"""

from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from app.core.llm_factory import LLMFactory
from app.services.analysis_prompts import AnalysisPrompts
from app.utils.logger import Logger

logger = Logger.get_logger(__name__)


def main():
    """Execute memory comparison with LLM-based analysis."""
    try:
        llm = LLMFactory.create_llm()

        # ---------------- TEXT DATA ----------------
        text_ml = """Machine learning is a branch of artificial intelligence that focuses on building systems that can learn from data without being explicitly programmed. It uses statistical techniques to improve performance over time. It is widely used in recommendation systems, fraud detection, and predictive analytics. Businesses rely on it for decision-making and automation. As data grows, machine learning becomes increasingly important in solving complex problems."""

        text_dl = """Deep learning is a subset of machine learning that uses neural networks with many layers. It is widely used in image recognition, speech processing, and NLP. These models require large datasets and high computational power. They automatically extract features from raw data. Deep learning continues to drive advancements in artificial intelligence across industries."""

        # ---------------- PROMPT ----------------
        prompt = PromptTemplate(
            input_variables=["history", "text"],
            template="""
You are a strict summarization assistant.

Conversation History:
{history}

Instructions:
- Always consider previous context if relevant
- Summarize in exactly 3 sentences
- Be concise and avoid repetition

Text:
{text}

Final Answer:
"""
        )

        # ---------------- BUFFER MEMORY ----------------
        memory_buffer = ConversationBufferMemory(
            memory_key="history",
            k=3,
            return_messages=False
        )

        chain_buffer = LLMChain(
            llm=llm,
            prompt=prompt,
            memory=memory_buffer,
            verbose=True
        )

        buffer_1 = chain_buffer.run(text=text_ml)
        buffer_2 = chain_buffer.run(text=text_dl)

        print("\n--- Buffer Memory ---\n")
        print(buffer_1)
        print("\n", buffer_2)

        # ---------------- SUMMARY MEMORY ----------------
        memory_summary = ConversationSummaryMemory(
            llm=llm,
            memory_key="history"
        )

        chain_summary = LLMChain(
            llm=llm,
            prompt=prompt,
            memory=memory_summary,
            verbose=True
        )

        summary_1 = chain_summary.run(text=text_ml)
        summary_2 = chain_summary.run(text=text_dl)

        print("\n--- Summary Memory ---\n")
        print(summary_1)
        print("\n", summary_2)

        # ---------------- LLM ANALYSIS ----------------
        logger.info("Generating memory comparison analysis")

        analysis_prompt = AnalysisPrompts.memory_comparison_analysis()
        analysis_chain = analysis_prompt | llm

        analysis_result = analysis_chain.invoke({
            "buffer_1": buffer_1,
            "buffer_2": buffer_2,
            "summary_1": summary_1,
            "summary_2": summary_2
        })

        print("\n--- LLM Analysis ---\n")
        print(analysis_result.content)

        logger.info("Task 6 completed successfully")

    except Exception as error:
        logger.error(f"Task 6 failed: {error}")
        raise


if __name__ == "__main__":
    main()