import os
from dotenv import load_dotenv
from datetime import datetime

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

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

splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = splitter.split_documents(documents)

vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

def retrieve_text(query):
    print("\n>>> TOOL CALLED: TextRetriever")
    print(">>> INPUT:", query)
    docs = retriever.get_relevant_documents(query)
    result = "\n".join([doc.page_content for doc in docs])
    print(">>> OUTPUT:", result[:200], "...\n")
    return result

retriever_tool = Tool(
    name="TextRetriever",
    func=retrieve_text,
    description="Retrieve relevant AI text from the document."
)

prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text in exactly 3 sentences:\n{text}"
)

chain = LLMChain(llm=llm, prompt=prompt)

def summarize_text(text):
    print("\n>>> TOOL CALLED: TextSummarizer")
    print(">>> INPUT:", text[:200], "...")
    result = chain.run(text)
    print(">>> OUTPUT:", result, "\n")
    return result

summarizer_tool = Tool(
    name="TextSummarizer",
    func=summarize_text,
    description="Summarize text into exactly 3 sentences."
)

def count_words(text):
    print("\n>>> TOOL CALLED: WordCounter")
    print(">>> INPUT:", text)
    result = f"Word count: {len(text.split())}"
    print(">>> OUTPUT:", result, "\n")
    return result

wordcount_tool = Tool(
    name="WordCounter",
    func=count_words,
    description="Count words in the given text."
)

def get_current_date(_):
    print("\n>>> TOOL CALLED: CurrentDate")
    result = datetime.now().strftime("%Y-%m-%d")
    print(">>> OUTPUT:", result, "\n")
    return result

date_tool = Tool(
    name="CurrentDate",
    func=get_current_date,
    description="Returns today's date."
)

def mock_web_search(query):
    print("\n>>> TOOL CALLED: WebSearch")
    print(">>> INPUT:", query)
    result = ("Recent AI updates highlight rapid progress in generative AI, "
              "large language models, and automation tools. Companies are investing "
              "in AI-driven solutions across industries, improving productivity, "
              "decision-making, and innovation while raising concerns about ethics "
              "and workforce impact.")
    print(">>> OUTPUT:", result, "\n")
    return result

search_tool = Tool(
    name="WebSearch",
    func=mock_web_search,
    description="Search for recent AI updates (mock data)."
)

agent = initialize_agent(
    tools=[retriever_tool, summarizer_tool, wordcount_tool, date_tool, search_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

text_input = """Artificial intelligence is transforming industries by improving efficiency and automation. 
It is widely used in healthcare for diagnosis, in finance for fraud detection, and in transportation for self-driving systems. 
AI also enhances customer experience through chatbots and recommendation systems, but raises concerns about privacy, bias, and job displacement."""

query1 = f"Summarize this text and tell me today's date:\n{text_input}"

response1 = agent.run(query1)

print("\n--- Test 1 Output ---\n")
print(response1)

query2 = "Summarize AI trends and search for recent updates."

response2 = agent.run(query2)

print("\n--- Test 2 Output ---\n")
print(response2)