import os
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever

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

splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)

docs = splitter.split_documents(documents)
vectorstore = FAISS.from_documents(docs, embeddings)

single_retriever = vectorstore.as_retriever()

query = "AI advancements"

single_docs = single_retriever.invoke(query)
single_text = "\n".join([doc.page_content for doc in single_docs])

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)

multi_docs = multi_retriever.invoke(query)
multi_text = "\n".join([doc.page_content for doc in multi_docs])

prompt = PromptTemplate.from_template(
    "Summarize the following text in exactly 3 sentences:\n{text}"
)

chain = prompt | llm

single_summary = chain.invoke({"text": single_text}).content
multi_summary = chain.invoke({"text": multi_text}).content

print("\n=========== SINGLE QUERY SUMMARY ===========\n")
print(single_summary)

print("\n=========== MULTI QUERY SUMMARY ===========\n")
print(multi_summary)

comparison_prompt = PromptTemplate.from_template("""
Compare these two summaries:

Single Query Summary:
{single}

Multi Query Summary:
{multi}

Explain which one is deeper and why.
Answer in 4-5 sentences.
""")

comparison_chain = comparison_prompt | llm

comparison = comparison_chain.invoke({
    "single": single_summary,
    "multi": multi_summary
}).content

print("\n=========== COMPARISON ===========\n")
print(comparison)