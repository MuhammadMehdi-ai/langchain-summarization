import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

load_dotenv()


llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("ENDPOINT_URL"),
    deployment_name=os.getenv("DEPLOYMENT_NAME"),
    api_version=os.getenv("API_VERSION"),
)

embeddings = AzureOpenAIEmbeddings(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("ENDPOINT_URL"),
    deployment=os.getenv("EMBEDDING_DEPLOYMENT_NAME"),
    api_version=os.getenv("API_VERSION"),
)

loader = TextLoader("ai_intro.txt")
documents = loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)

docs = text_splitter.split_documents(documents)
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()
query = "AI milestones"
retrieved_docs = retriever.invoke(query)
retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])
prompt = PromptTemplate.from_template(
    "Summarize the following text in 3 sentences:\n{text}"
)
chain = prompt | llm
result = chain.invoke({"text": retrieved_text})
print("\n Retrieved Text:\n", retrieved_text)
print("\nSummary:\n", result.content)