import os
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

load_dotenv()

AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
ENDPOINT = os.getenv("ENDPOINT_URL")
DEPLOYMENT = os.getenv("DEPLOYMENT_NAME")
API_VERSION = os.getenv("API_VERSION")
EMBEDDING_DEPLOYMENT = os.getenv("EMBEDDING_DEPLOYMENT_NAME")

if not AZURE_API_KEY:
    raise ValueError("Missing AZURE_OPENAI_API_KEY in .env")

llm = AzureChatOpenAI(
    api_key=AZURE_API_KEY,
    azure_endpoint=ENDPOINT,
    deployment_name=DEPLOYMENT,
    api_version=API_VERSION,
)

embeddings = AzureOpenAIEmbeddings(
    api_key=AZURE_API_KEY,
    azure_endpoint=ENDPOINT,
    deployment=EMBEDDING_DEPLOYMENT,
    api_version=API_VERSION,
)

PDF_PATH = "task-7/ai_ethics.pdf"

if not os.path.exists(PDF_PATH):
    raise FileNotFoundError("ai_ethics.pdf not found. Please add a 2-page AI ethics PDF.")

print("\nLoading PDF...")
pdf_loader = PyPDFLoader(PDF_PATH)
pdf_docs = pdf_loader.load()

print("Loading Web Page...")
WEB_URL = "https://www.ibm.com/blog/artificial-intelligence-trends/"
web_loader = WebBaseLoader(WEB_URL)
web_docs = web_loader.load()

print(f"PDF Pages Loaded: {len(pdf_docs)}")
print(f"Web Docs Loaded: {len(web_docs)}")


splitter = CharacterTextSplitter(
    chunk_size=150,   
    chunk_overlap=30  
)

pdf_chunks = splitter.split_documents(pdf_docs)
web_chunks = splitter.split_documents(web_docs)

print(f"\nPDF Chunks: {len(pdf_chunks)}")
print(f"Web Chunks: {len(web_chunks)}")

print("\nCreating FAISS Vector Stores...")

pdf_vectorstore = FAISS.from_documents(pdf_chunks, embeddings)
web_vectorstore = FAISS.from_documents(web_chunks, embeddings)

pdf_retriever = pdf_vectorstore.as_retriever()
web_retriever = web_vectorstore.as_retriever()

query = "AI challenges"

print(f"\nQuery: {query}")

pdf_results = pdf_retriever.invoke(query)
web_results = web_retriever.invoke(query)

pdf_text = "\n".join([doc.page_content for doc in pdf_results])
web_text = "\n".join([doc.page_content for doc in web_results])

summary_prompt = PromptTemplate.from_template(
    "Summarize the following text in exactly 3 sentences:\n{text}"
)

summary_chain = RunnableSequence(summary_prompt | llm)

pdf_summary = summary_chain.invoke({"text": pdf_text}).content
web_summary = summary_chain.invoke({"text": web_text}).content

print("\nPDF SUMMARY\n")
print(pdf_summary)

print("\nWEB SUMMARY\n")
print(web_summary)

comparison_prompt = PromptTemplate.from_template("""
Compare the quality of these two summaries:

PDF Summary:
{pdf_summary}

Web Summary:
{web_summary}

Explain differences in:
1. Clarity
2. Detail
3. Structure
4. Noise or irrelevant content

Answer in 5-6 sentences.
""")

comparison_chain = RunnableSequence(comparison_prompt | llm)

comparison = comparison_chain.invoke({
    "pdf_summary": pdf_summary,
    "web_summary": web_summary
}).content

print("\nCOMPARISON\n")
print(comparison)
