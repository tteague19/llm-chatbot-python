""""Implements a LangChain Agent."""
from typing import Sequence, Optional

from langchain.agents import AgentType, initialize_agent, AgentExecutor
from langchain.schema import BaseMemory
from langchain.schema.language_model import BaseLanguageModel
from langchain.tools import BaseTool


def create_agent(
        tools: Sequence[BaseTool],
        llm: BaseLanguageModel,
        memory: BaseMemory,
        agent_type: AgentType,
        system_message: str,
        verbose: bool = True,
) -> AgentExecutor:
    """
    Create a LangChain Agent using specified arguments.

    :param tools: A sequence of tools to which the agent has access
    :type tools: Sequence[BaseTool]
    :param llm: A large language model for the agent to use
    :type llm: BaseLanguageModel
    :param memory: A memory context for the agent to use
    :type memory: BaseMemory
    :param agent_type: The type of agent to create
    :type agent_type: AgentType
    :param system_message: The system message for the agent to use
    :type system_message: str
    :param verbose: Whether to include additional diagnostic output in the
        return from the agent, defaults to True
    :type verbose: bool
    :return: An agent executor object
    :rtype: AgentExecutor
    """
    return initialize_agent(
        tools=tools,
        llm=llm,
        memory=memory,
        verbose=verbose,
        agent=agent_type,
        agent_kwargs={"system_message": system_message},
    )


def generate_response_from_agent(
        prompt: str, agent: AgentExecutor, extraction_key: str,
) -> Optional[str]:
    """
    Generate a response from a user prompt using a LangChain Agent.

    :param prompt: A prompt for the :param:`agent` to use in generating a
        response
    :type prompt: str
    :param agent: An agent executor object
    :type agent: AgentExecutor
    :param extraction_key: The key in the dictionary that the :param:`agent`
        returns after generating a response to the :param:`prompt`
    :type extraction_key: str
    :return: The value at :param:`extraction_key` after the :param:`agent`
        generates a response to the :param:`prompt`
    :rtype: str
    """
    response = agent(prompt)
    return response.get(extraction_key, None)
