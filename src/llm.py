"""Implements an LLM for use in the Streamlit application."""
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings


def create_embedding_model(
        api_key: str, model_name: str = "text-embedding-ada-002",
) -> OpenAIEmbeddings:
    """
    Create an embedding model.

    :param api_key: An API key to enable the use of a proprietary model
    :type api_key: str
    :param model_name: The name of the embedding model to use, defaults to
        "text-embedding-ada-002"
    :return: An embedding model
    :rtype: OpenAIEmbeddings
    """
    return OpenAIEmbeddings(openai_api_key=api_key, model=model_name)


def create_chat_llm(
        api_key: str, model_name: str = "gpt-3.5-turbo",
) -> ChatOpenAI:
    """
    Create a large language model for use in a chat-based context.

    :param api_key: An API key to enable the use of a proprietary model
    :type api_key: str
    :param model_name: The name of the embedding model to use, defaults to
        "gpt-3.5-turbo"
    :type model_name: str
    :return: A chat-based large language model
    :rtype: ChatOpenAI
    """
    return ChatOpenAI(model_name=model_name, openai_api_key=api_key)
