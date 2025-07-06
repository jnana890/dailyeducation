# models/llama3/llm.py

from langchain_community.llms import Ollama


def get_llama3_llm(model_name: str = "llama3", temperature: float = 0.1):
    """
    Load a local LLaMA 3 model using Ollama.

    Args:
        model_name (str): Name of the model pulled via Ollama (e.g., 'llama3', 'mistral').
        temperature (float): Temperature setting for generation.

    Returns:
        Ollama: LangChain-compatible local LLM instance.
    """
    return Ollama(model=model_name, temperature=temperature)
