# app/web_agent.py

from langchain.agents import initialize_agent, AgentType, Tool
from langchain.llms import Ollama
from utils.web_search import search_web  # your Serper function

def get_llama3_llm():
    return Ollama(model="llama3")

def get_fallback_answer(query: str) -> str:
    """
    Uses an agent with web search tool to answer web-based questions.
    """
    tools = [
        Tool(
            name="WebSearch",
            func=search_web,
            description="Useful for answering questions about current events or real-time information.",
        )
    ]

    agent = initialize_agent(
        tools=tools,
        llm=get_llama3_llm(),
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True  # ✅ Fix for OUTPUT_PARSING_FAILURE
    )

    return agent.invoke(query)  # ✅ Avoid deprecated .run()
