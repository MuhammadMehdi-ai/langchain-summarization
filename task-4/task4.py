import os
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

load_dotenv()

# -------- LLM --------
llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("ENDPOINT_URL"),
    deployment_name=os.getenv("DEPLOYMENT_NAME"),
    api_version=os.getenv("API_VERSION"),
)

# -------- Summarization Chain --------
prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text in exactly 3 sentences:\n{text}"
)

chain = LLMChain(llm=llm, prompt=prompt)

# -------- Tool --------
def summarize_text(text):
    return chain.run(text)

tool = Tool(
    name="TextSummarizer",
    func=summarize_text,
    description="Summarizes given text into 3 sentences."
)

# -------- Agent --------
agent = initialize_agent(
    tools=[tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# -------- Test 1 --------
text = """Artificial intelligence is transforming healthcare by improving diagnosis,
personalized treatment, and automation. AI helps in medical imaging, drug discovery,
and patient monitoring. However, concerns like privacy, bias, and ethics remain."""

response = agent.run(f"Summarize the impact of AI on healthcare:\n{text}")
print("\nResult 1:\n", response)

# -------- Test 2 --------
response2 = agent.run("Summarize something interesting")
print("\nResult 2:\n", response2)