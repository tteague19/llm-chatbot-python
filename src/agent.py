""""Implements a LangChain Agent."""
from typing import Sequence

from langchain.agents import AgentType, initialize_agent, AgentExecutor
from langchain.schema import BaseMemory
from langchain.schema.language_model import BaseLanguageModel
from langchain.tools import BaseTool


def create_agent(
        tools: Sequence[BaseTool],
        llm: BaseLanguageModel,
        memory: BaseMemory,
        agent_type: AgentType,
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
    :param verbose: Whether to include additional diagnostic output in the
        return from the agent, defaults to True
    :type verbose: bool
    :return: An agent executor object
    :rtype: AgentExecutor
    """
    return initialize_agent(
        tools=tools, llm=llm, memory=memory, verbose=verbose, agent=agent_type,
    )
