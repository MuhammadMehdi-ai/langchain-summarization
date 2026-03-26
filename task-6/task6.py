# import os
# from dotenv import load_dotenv

# from langchain_openai import AzureChatOpenAI
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
# from langchain.agents import initialize_agent, AgentType
# from langchain.tools import Tool

# load_dotenv()

# # LLM 
# llm = AzureChatOpenAI(
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#     azure_endpoint=os.getenv("ENDPOINT_URL"),
#     deployment_name=os.getenv("DEPLOYMENT_NAME"),
#     api_version=os.getenv("API_VERSION"),
# )

# # Prompt 
# prompt = PromptTemplate(
#     input_variables=["history", "text"],
#     template="""
# Use previous context if relevant.

# Summarize the following text in exactly 3 sentences:
# {text}
# """
# )

# #  Texts 
# text_ml = """Machine learning is a field of computer science that focuses on building systems that can learn from data without being explicitly programmed. It relies on algorithms that detect patterns and improve performance over time. Machine learning is commonly used in recommendation systems, spam filtering, fraud detection, and predictive analytics. Businesses use it to make better decisions based on data insights. It also plays a role in automation and personalization in modern applications. As data continues to grow, machine learning is becoming increasingly important across industries for solving complex problems efficiently."""

# text_dl = """Deep learning is a powerful technology used to train artificial neural networks with many layers. It is widely applied in image recognition, speech processing, and language translation systems. Deep learning models require large amounts of data and high computational power to perform effectively. These models are inspired by the human brain and can automatically extract features from raw data. They are used in applications like self-driving cars, virtual assistants, and medical image analysis. Deep learning continues to advance rapidly, enabling breakthroughs in artificial intelligence and improving the performance of complex systems."""

# # Buffer Memory + Agent

# print("\n" + "="*60)
# print("USING ConversationBufferMemory (WITH AGENT LOGS)")
# print("="*60)

# memory_buffer = ConversationBufferMemory(
#     memory_key="history",
#     return_messages=False
# )

# chain_buffer = LLMChain(llm=llm, prompt=prompt, memory=memory_buffer)

# # Tool
# def summarize_tool(text):
#     return chain_buffer.run(text=text)

# tool = Tool(
#     name="TextSummarizer",
#     func=summarize_tool,
#     description="Summarizes text into exactly 3 sentences."
# )

# # Agent with verbose logs
# agent = initialize_agent(
#     tools=[tool],
#     llm=llm,
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True
# )

# # Run agent
# print("\n--- Step 1: Machine Learning ---")
# agent.run(f"Summarize this text:\n{text_ml}")

# print("\n--- Step 2: Deep Learning (uses memory) ---")
# agent.run(f"Summarize this text:\n{text_dl}")


# # Summary Memory + Agent

# print("\n" + "="*60)
# print("USING ConversationSummaryMemory (WITH AGENT LOGS)")
# print("="*60)

# memory_summary = ConversationSummaryMemory(
#     llm=llm,
#     memory_key="history"
# )

# chain_summary = LLMChain(llm=llm, prompt=prompt, memory=memory_summary)

# def summarize_tool_summary(text):
#     return chain_summary.run(text=text)

# tool_summary = Tool(
#     name="TextSummarizer",
#     func=summarize_tool_summary,
#     description="Summarizes text into exactly 3 sentences."
# )

# agent_summary = initialize_agent(
#     tools=[tool_summary],
#     llm=llm,
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True
# )

# print("\n--- Step 1: Machine Learning ---")
# agent_summary.run(f"Summarize this text:\n{text_ml}")

# print("\n--- Step 2: Deep Learning (uses summarized memory) ---")
# agent_summary.run(f"Summarize this text:\n{text_dl}")


import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

load_dotenv()
llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("ENDPOINT_URL"),
    deployment_name=os.getenv("DEPLOYMENT_NAME"),
    api_version=os.getenv("API_VERSION"),
    temperature=0  
)


prompt = PromptTemplate(
    input_variables=["history", "text"],
    template="""
You are a STRICT summarization assistant.

Conversation History (Context):
{history}

Instructions:
- ALWAYS consider previous context if relevant.
- Summarize the following text in exactly 3 sentences:.
- Do NOT exceed or reduce sentence count.
- Keep the summary concise, factual, and non-repetitive.
- If prior context helps, integrate it meaningfully.

Text to summarize:
{text}

Final Answer (ONLY 3 sentences):
"""
)
text_ml = """Machine learning is a field of computer science that focuses on building systems that can learn from data without being explicitly programmed. It relies on algorithms that detect patterns and improve performance over time. Machine learning is commonly used in recommendation systems, spam filtering, fraud detection, and predictive analytics. Businesses use it to make better decisions based on data insights. It also plays a role in automation and personalization in modern applications. As data continues to grow, machine learning is becoming increasingly important across industries for solving complex problems efficiently."""

text_dl = """Deep learning is a powerful technology used to train artificial neural networks with many layers. It is widely applied in image recognition, speech processing, and language translation systems. Deep learning models require large amounts of data and high computational power to perform effectively. These models are inspired by the human brain and can automatically extract features from raw data. They are used in applications like self-driving cars, virtual assistants, and medical image analysis. Deep learning continues to advance rapidly, enabling breakthroughs in artificial intelligence and improving the performance of complex systems."""



print("USING ConversationBufferMemory (RAW FULL CONTEXT)")

memory_buffer = ConversationBufferMemory(
    memory_key="history",
    return_messages=False
)

chain_buffer = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory_buffer,
    verbose=True
)

# TOOL
def summarize_tool(text):
    print("\nTOOL CALLED (Buffer Memory)")
    return chain_buffer.run(text=text)

tool = Tool(
    name="TextSummarizer",
    func=summarize_tool,
    description="Summarizes any text into exactly 3 sentences using context."
)

# AGENT
agent = initialize_agent(
    tools=[tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={
        "prefix": """You are an AI agent.

STRICT RULES:
- ALWAYS use the TextSummarizer tool for summarization.
- NEVER summarize by yourself.
- DO NOT answer directly.
"""
    }
)

# RUN STEP 1
print("\nStep 1: Machine Learning")
response1 = agent.run(text_ml)
print("\n✅ Final Output:\n", response1)

print("\nMemory After Step 1:")
print(memory_buffer.load_memory_variables({}))

# RUN STEP 2
print("\nStep 2: Deep Learning (uses previous context)")
response2 = agent.run(text_dl)
print("\n✅ Final Output:\n", response2)

print("\nMemory After Step 2:")
print(memory_buffer.load_memory_variables({}))


# PART 2: ConversationSummaryMemory (COMPRESSED MEMORY)

print("USING ConversationSummaryMemory (COMPRESSED CONTEXT)")

memory_summary = ConversationSummaryMemory(
    llm=llm,
    memory_key="history"
)

chain_summary = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory_summary,
    verbose=True
)

def summarize_tool_summary(text):
    print("\nTOOL CALLED (Summary Memory)")
    return chain_summary.run(text=text)

tool_summary = Tool(
    name="TextSummarizer",
    func=summarize_tool_summary,
    description="Summarizes text into exactly 3 sentences using summarized context."
)

agent_summary = initialize_agent(
    tools=[tool_summary],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={
        "prefix": """You are an AI agent.

STRICT RULES:
- ALWAYS use the TextSummarizer tool
- NEVER answer directly
"""
    }
)

# RUN STEP 1
print("\nStep 1: Machine Learning")
response3 = agent_summary.run(text_ml)
print("\nFinal Output:\n", response3)

print("\nSummary Memory After Step 1:")
print(memory_summary.load_memory_variables({}))

# RUN STEP 2
print("\nStep 2: Deep Learning (uses summarized context)")
response4 = agent_summary.run(text_dl)
print("\nFinal Output:\n", response4)

print("\nSummary Memory After Step 2:")
print(memory_summary.load_memory_variables({}))