import os
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

load_dotenv()

# -------- LLM --------
llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("ENDPOINT_URL"),
    deployment_name=os.getenv("DEPLOYMENT_NAME"),
    api_version=os.getenv("API_VERSION"),
)

# -------- Embeddings --------
embeddings = AzureOpenAIEmbeddings(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("ENDPOINT_URL"),
    deployment=os.getenv("EMBEDDING_DEPLOYMENT_NAME"),
    api_version=os.getenv("API_VERSION"),
)

loader = TextLoader("ai_intro.txt")
documents = loader.load()

splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = splitter.split_documents(documents)

vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

# Tool 1: Retriever
def retrieve_text(query):
    docs = retriever.get_relevant_documents(query)
    return "\n".join([doc.page_content for doc in docs])

retriever_tool = Tool(
    name="TextRetriever",
    func=retrieve_text,
    description="Use this tool to retrieve relevant text from the AI document based on a query."
)

# Tool 2: Summarizer 
prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text in exactly 3 sentences:\n{text}"
)

chain = LLMChain(llm=llm, prompt=prompt)

def summarize_text(text):
    return chain.run(text)

summarizer_tool = Tool(
    name="TextSummarizer",
    func=summarize_text,
    description="Use this tool to summarize provided text into exactly 3 sentences."
)

# Tool 3: Word Counter 
def count_words(text):
    return f"Word count: {len(text.split())}"

wordcount_tool = Tool(
    name="WordCounter",
    func=count_words,
    description="Use this tool AFTER summarizing to count the number of words in the summary."
)

# -------- Agent --------
agent = initialize_agent(
    tools=[retriever_tool, summarizer_tool, wordcount_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# -------- Test Query --------
query = """
Step 1: Retrieve relevant text about AI breakthroughs from the document.
Step 2: Summarize the retrieved text in exactly 3 sentences.
Step 3: Use the WordCounter tool to count the number of words in the summary.
Return both the summary and the word count.
"""

response = agent.run(query)

print("\n--- Final Output ---\n", response)