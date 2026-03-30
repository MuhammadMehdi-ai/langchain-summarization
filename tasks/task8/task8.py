"""
Task 8: Structured output parsing (final version with LLM analysis).
"""

from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate
from app.core.llm_factory import LLMFactory
from app.services.analysis_prompts import AnalysisPrompts
from app.utils.logger import Logger

logger = Logger.get_logger(__name__)


def main():
    """Execute structured output parsing."""
    try:
        llm = LLMFactory.create_llm()

        # ---------------- SCHEMA ----------------
        schemas = [
            ResponseSchema(
                name="summary",
                description="Summary of the input text in exactly 3 sentences"
            ),
            ResponseSchema(
                name="length",
                description="Character count of the summary"
            )
        ]

        parser = StructuredOutputParser.from_response_schemas(schemas)

        # ---------------- PROMPT ----------------
        prompt = PromptTemplate(
            input_variables=["text"],
            partial_variables={
                "format_instructions": parser.get_format_instructions()
            },
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

        # ---------------- INPUT TEXT (~150 WORDS) ----------------
        text = """
Artificial intelligence is widely applied across industries to enhance efficiency and decision-making. 
In healthcare, AI assists in diagnosing diseases, analyzing medical images, and personalizing treatments. 
In finance, it is used for fraud detection, risk assessment, and algorithmic trading. 
AI also plays a key role in transportation through autonomous vehicles and traffic optimization. 
In customer service, AI-powered chatbots improve user experience by providing instant responses. 
Despite its advantages, AI introduces challenges such as data privacy concerns, algorithmic bias, and job displacement. 
Organizations must adopt ethical AI practices to ensure fairness, transparency, and accountability while leveraging its benefits.
"""

        result = chain.invoke({"text": text})

        # ---------------- OUTPUT ----------------
        print("\n--- STRUCTURED OUTPUT ---\n")
        print(result)

        print("\nSummary:\n", result["summary"])
        print("\nLength:\n", result["length"])

        # ---------------- LLM ANALYSIS ----------------
        analysis_prompt = AnalysisPrompts.structured_output_analysis()
        analysis_chain = analysis_prompt | llm

        analysis_result = analysis_chain.invoke({
            "output": result
        })

        print("\n--- LLM ANALYSIS ---\n")
        print(analysis_result.content)

        logger.info("Task 8 completed successfully")

    except Exception as error:
        logger.error(f"Task 8 failed: {error}")
        raise


if __name__ == "__main__":
    main()