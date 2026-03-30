"""
Agent service wrapper.
"""

from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from app.core.llm_factory import LLMFactory


class AgentService:
    """Encapsulates agent creation and execution."""

    def __init__(self, tools: list):
        self.llm = LLMFactory.create_llm()
        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )

    def run(self, query: str) -> str:
        """Execute agent query."""
        return self.agent.run(query)