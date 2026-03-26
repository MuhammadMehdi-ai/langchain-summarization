import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()
llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("ENDPOINT_URL"),
    deployment_name=os.getenv("DEPLOYMENT_NAME"),
    api_version=os.getenv("API_VERSION"),
)

text = """Artificial intelligence (AI) refers to the simulation of human intelligence
in machines that are programmed to think and learn like humans. 
AI can be categorized into narrow AI, which is designed for specific
tasks, and general AI, which has the ability to perform any 
intellectual task that a human can do. The development of AI has the 
potential to revolutionize various industries, but it also raises 
ethical concerns regarding privacy, job displacement, 
and decision-making transparency."""

# 3 Sentence Summary 
prompt_3 = PromptTemplate.from_template(
    "Summarize the following text in exactly 3 sentences:\n{text}"
)

chain_3 = prompt_3 | llm

result_3 = chain_3.invoke({"text": text})
print("3 Sentence Summary:\n", result_3.content)


# 1 Sentence Summary
prompt_1 = PromptTemplate.from_template(
    "Summarize the following text in exactly 1 sentence:\n{text}"
)

chain_1 = prompt_1 | llm

result_1 = chain_1.invoke({"text": text})
print("\n1 Sentence Summary:\n", result_1.content)