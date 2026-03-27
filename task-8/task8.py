import os
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("ENDPOINT_URL"),
    deployment_name=os.getenv("DEPLOYMENT_NAME"),
    api_version=os.getenv("API_VERSION"),
)


response_schemas = [
    ResponseSchema(
        name="summary",
        description="Summary of the input text"
    ),
    ResponseSchema(
        name="length",
        description="Character count of the summary"
    ),
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)


format_instructions = parser.get_format_instructions()


prompt = PromptTemplate(
    input_variables=["text"],
    partial_variables={"format_instructions": format_instructions},
    template="""
Summarize the following text in exactly 3 sentences.

{text}

Return output in JSON format with:
- summary
- length (character count)

{format_instructions}
"""
)


chain = prompt | llm | parser

text = """Artificial intelligence (AI) is widely applied across industries to enhance efficiency and decision-making. 
In healthcare, AI assists in diagnosing diseases, analyzing medical images, and personalizing treatments. 
In finance, it is used for fraud detection, risk assessment, and algorithmic trading. 
AI also plays a key role in transportation through autonomous vehicles and traffic optimization. 
In customer service, AI-powered chatbots improve user experience by providing instant responses. 
Despite its advantages, AI introduces challenges such as data privacy concerns, algorithmic bias, and job displacement. 
Organizations must adopt ethical AI practices to ensure fairness, transparency, and accountability while leveraging its benefits."""


result = chain.invoke({"text": text})

print("\nSTRUCTURED OUTPUT (JSON):\n")
print(result)

print("\nSummary:\n", result["summary"])
print("\nLength: ", result["length"])