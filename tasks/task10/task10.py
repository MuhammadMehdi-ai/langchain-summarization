"""
Task 10: QA on summary vs full text (complete implementation).
"""

from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from app.core.llm_factory import LLMFactory
from app.services.summarizer_service import SentenceSummarizer
from app.services.analysis_prompts import AnalysisPrompts
from app.utils.logger import Logger

logger = Logger.get_logger(__name__)


def main():
    """Compare QA performance on summary vs full text."""
    try:
        llm = LLMFactory.create_llm()

        # ---------------- LOAD TEXT ----------------
        loader = TextLoader("ai_intro.txt")
        documents = loader.load()

        full_text = "\n".join([doc.page_content for doc in documents])

        # ---------------- SUMMARIZATION ----------------
        summarizer = SentenceSummarizer(3)
        summary = summarizer.summarize(full_text)

        print("\n--- SUMMARY ---\n")
        print(summary)

        # ---------------- QA PROMPT ----------------
        qa_prompt = PromptTemplate.from_template("""
Answer the question based only on the text below.

Text:
{text}

Question:
{question}

Answer clearly and concisely.
""")

        qa_chain = qa_prompt | llm

        question = "What’s the key event mentioned?"

        # ---------------- QA ON SUMMARY ----------------
        summary_answer = qa_chain.invoke({
            "text": summary,
            "question": question
        }).content

        print("\n--- ANSWER FROM SUMMARY ---\n")
        print(summary_answer)

        # ---------------- QA ON FULL TEXT ----------------
        full_answer = qa_chain.invoke({
            "text": full_text,
            "question": question
        }).content

        print("\n--- ANSWER FROM FULL TEXT ---\n")
        print(full_answer)

        # ---------------- LLM COMPARISON ----------------
        analysis_prompt = AnalysisPrompts.qa_comparison_analysis()
        analysis_chain = analysis_prompt | llm

        comparison = analysis_chain.invoke({
            "summary_ans": summary_answer,
            "full_ans": full_answer
        })

        print("\n--- COMPARISON ---\n")
        print(comparison.content)

        logger.info("Task 10 completed successfully")

    except Exception as error:
        logger.error(f"Task 10 failed: {error}")
        raise


if __name__ == "__main__":
    main()