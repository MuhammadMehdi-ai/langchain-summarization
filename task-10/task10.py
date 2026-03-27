import os
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate

load_dotenv()

AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
ENDPOINT = os.getenv("ENDPOINT_URL")
DEPLOYMENT = os.getenv("DEPLOYMENT_NAME")
API_VERSION = os.getenv("API_VERSION")

llm = AzureChatOpenAI(
    api_key=AZURE_API_KEY,
    azure_endpoint=ENDPOINT,
    deployment_name=DEPLOYMENT,
    api_version=API_VERSION,
)

loader = TextLoader("ai_intro.txt")
documents = loader.load()

full_text = "\n".join([doc.page_content for doc in documents])

summary_prompt = PromptTemplate.from_template(
    "Summarize the following text in exactly 3 sentences:\n{text}"
)

summary_chain = summary_prompt | llm

summary = summary_chain.invoke({"text": full_text}).content

print("\n=========== SUMMARY ===========\n")
print(summary)

qa_prompt = PromptTemplate.from_template(
    """
Answer the question based on the text below.

Text:
{text}

Question:
{question}

Answer clearly and concisely.
"""
)

qa_chain = qa_prompt | llm

question = "What’s the key event mentioned?"

summary_answer = qa_chain.invoke({
    "text": summary,
    "question": question
}).content

full_answer = qa_chain.invoke({
    "text": full_text,
    "question": question
}).content

print("\n=========== ANSWER FROM SUMMARY ===========\n")
print(summary_answer)

print("\n=========== ANSWER FROM FULL TEXT ===========\n")
print(full_answer)

comparison_prompt = PromptTemplate.from_template(
    """Compare these two answers:

Answer from Summary:
{summary_ans}

Answer from Full Text:
{full_ans}

Evaluate:
1. Which is more concise?
2. Which is more accurate?
3. Why?

Answer in 4-5 sentences."""
)

comparison_chain = comparison_prompt | llm

comparison = comparison_chain.invoke({
    "summary_ans": summary_answer,
    "full_ans": full_answer
}).content

print("\n=========== COMPARISON ===========\n")
print(comparison)