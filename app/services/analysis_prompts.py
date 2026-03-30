"""
Reusable prompts for agent behavior and pipeline analysis.
"""

from langchain_core.prompts import PromptTemplate


class AnalysisPrompts:
    """Factory class for analysis prompts."""

    @staticmethod
    def agent_behavior_analysis():
        """Task 4 analysis prompt."""
        return PromptTemplate.from_template("""
You are an expert evaluator of AI agent behavior.

Two outputs were generated:

Test 1 Output:
{test1}

Test 2 Output:
{test2}

Analyze the behavior of the agent:

1. Did the agent correctly follow instructions in Test 1?
2. What happened in Test 2 with the vague query?
3. Did the agent hallucinate or generate its own input?
4. What are the risks of this behavior?
5. How can this be improved in production systems?

Answer in 6–8 concise sentences.
""")

    @staticmethod
    def pipeline_analysis():
        """Task 5 pipeline analysis prompt."""
        return PromptTemplate.from_template("""
You are an expert evaluator of an AI pipeline involving retrieval, summarization, and tool usage.

Two outputs were generated:

Test 1 Output:
{test1}

Test 2 Output:
{test2}

Analyze the pipeline execution:

1. Did the agent correctly use the retriever tool?
2. Did the summarizer produce a valid 3-sentence summary?
3. Was the word count step executed correctly?
4. Did the agent follow the multi-step instructions properly?
5. Identify any inefficiencies or mistakes in tool usage.
6. Suggest improvements for making the pipeline more reliable.

Answer in 6–8 concise sentences.
""")
        
    @staticmethod
    def memory_comparison_analysis():
        """Task 6 memory comparison prompt."""
        return PromptTemplate.from_template("""
    You are an expert in evaluating memory mechanisms in LLM systems.

    Two memory strategies were used:

    --- Buffer Memory Output ---
    First Summary:
    {buffer_1}

    Second Summary:
    {buffer_2}

    --- Summary Memory Output ---
    First Summary:
    {summary_1}

    Second Summary:
    {summary_2}

    Analyze the differences:

    1. How does BufferMemory affect the second summary?
    2. How does SummaryMemory affect the second summary?
    3. Which one preserves context better?
    4. Which one is more efficient?
    5. What are the trade-offs between both approaches?
    6. When should each memory type be used?

    Answer in 6–8 concise sentences.
    """)
    @staticmethod
    def document_comparison_analysis():
        """Task 7 PDF vs Web comparison prompt."""
        return PromptTemplate.from_template("""
    You are an expert evaluator of document retrieval and summarization quality.

    Two summaries were generated from different sources:

    PDF Summary:
    {pdf_summary}

    Web Summary:
    {web_summary}

    Analyze and compare:

    1. Which summary is more clear and structured?
    2. Which one provides more detailed information?
    3. Which one contains more noise or irrelevant content?
    4. How does source type (PDF vs Web) affect quality?
    5. Which summary is more reliable and why?
    6. Suggest improvements for better retrieval or summarization.

    Answer in 6–8 concise sentences.
    """)
    @staticmethod
    def structured_output_analysis():
        """Task 8 structured output evaluation."""
        return PromptTemplate.from_template("""
    You are an expert evaluator of structured LLM outputs.

    Given the following JSON output:

    {output}

    Evaluate:

    1. Is the output valid JSON?
    2. Does it contain both "summary" and "length" fields?
    3. Is the summary exactly 3 sentences?
    4. Is the length correctly representing character count?
    5. Are there any formatting issues?

    Answer in 5–6 concise sentences.
    """)
    @staticmethod
    def multi_query_analysis():
        """Task 9 comparison analysis."""
        return PromptTemplate.from_template("""
    You are an expert evaluator of retrieval quality.

    Compare the following two summaries:

    Single Query Summary:
    {single}

    Multi Query Summary:
    {multi}

    Evaluate:

    1. Which summary is more detailed?
    2. Which captures broader context?
    3. Which is more accurate?
    4. Why multi-query retrieval improves or does not improve results?

    Answer in 4–5 concise sentences.
    """)
    @staticmethod
    def qa_comparison_analysis():
        """Task 10 QA comparison analysis."""
        return PromptTemplate.from_template("""
    You are an expert evaluator.

    Compare the following two answers:

    Answer from Summary:
    {summary_ans}

    Answer from Full Text:
    {full_ans}

    Evaluate:

    1. Which is more concise?
    2. Which is more accurate?
    3. Why?

    Answer in 4–5 concise sentences.
    """)
