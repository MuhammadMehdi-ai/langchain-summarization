"""
Task 7: PDF + Web loading, retrieval, and comparison (LLM analysis).
"""

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from app.core.embeddings_factory import EmbeddingsFactory
from app.services.summarizer_service import SentenceSummarizer
from app.core.llm_factory import LLMFactory
from app.services.analysis_prompts import AnalysisPrompts
from app.utils.logger import Logger

logger = Logger.get_logger(__name__)


def main():
    """Load, split, retrieve, summarize, and compare using LLM."""
    try:
        # ---------------- LOAD ----------------
        pdf_loader = PyPDFLoader("tasks/task7/ai_ethics.pdf")
        web_loader = WebBaseLoader("https://www.ibm.com/blog/artificial-intelligence-trends/")

        pdf_docs = pdf_loader.load()
        web_docs = web_loader.load()

        # ---------------- SPLIT ----------------
        splitter = CharacterTextSplitter(chunk_size=150, chunk_overlap=30)

        pdf_chunks = splitter.split_documents(pdf_docs)
        web_chunks = splitter.split_documents(web_docs)

        # ---------------- EMBEDDINGS ----------------
        embeddings = EmbeddingsFactory.create_embeddings()

        pdf_store = FAISS.from_documents(pdf_chunks, embeddings)
        web_store = FAISS.from_documents(web_chunks, embeddings)

        pdf_retriever = pdf_store.as_retriever()
        web_retriever = web_store.as_retriever()

        # ---------------- QUERY ----------------
        query = "AI challenges"

        pdf_docs = pdf_retriever.invoke(query)
        web_docs = web_retriever.invoke(query)

        pdf_text = "\n".join([d.page_content for d in pdf_docs])
        web_text = "\n".join([d.page_content for d in web_docs])

        # ---------------- SUMMARIZE ----------------
        summarizer = SentenceSummarizer(3)

        pdf_summary = summarizer.summarize(pdf_text)
        web_summary = summarizer.summarize(web_text)

        print("\n--- PDF SUMMARY ---\n", pdf_summary)
        print("\n--- WEB SUMMARY ---\n", web_summary)

        # ---------------- LLM COMPARISON ----------------
        llm = LLMFactory.create_llm()

        analysis_prompt = AnalysisPrompts.document_comparison_analysis()
        analysis_chain = analysis_prompt | llm

        analysis_result = analysis_chain.invoke({
            "pdf_summary": pdf_summary,
            "web_summary": web_summary
        })

        print("\n--- LLM ANALYSIS ---\n")
        print(analysis_result.content)

        logger.info("Task 7 completed successfully")

    except Exception as error:
        logger.error(f"Task 7 failed: {error}")
        raise


if __name__ == "__main__":
    main()