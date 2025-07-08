from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain.tools import Tool

from src.retrieval_generation import generate_response, get_llm
from src.retriever import get_hybrid_retriever

# This is the prompt that instructs the agent on how to behave.
AGENT_PROMPT_TEMPLATE = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""

def create_rag_agent(retriever, llm):
    """Creates and returns the RAG agent executor."""

    # 1. Define the tool for the agent
    # This wraps our entire advanced RAG pipeline into a single tool the agent can call.
    advanced_rag_tool = Tool(
        name="AdvancedRAGSearch",
        func=lambda query: generate_response(query, retriever, llm, get_rag_chain(llm)),
        description="Use this tool to find the answer to a user's question from a set of documents. The input should be the user's question.",
    )

    tools = [advanced_rag_tool]

    # 2. Create the agent prompt
    prompt = PromptTemplate.from_template(AGENT_PROMPT_TEMPLATE)

    # 3. Create the agent itself
    agent = create_react_agent(llm, tools, prompt)

    # 4. Create the agent executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    print("RAG Agent created successfully.")
    return agent_executor

# We need to import get_rag_chain here to avoid circular dependency issues at runtime
from src.retrieval_generation import get_rag_chain
