"""Implements the Streamlit frontend for the chatbot."""
from time import sleep

import streamlit as st
from langchain.agents import AgentType, AgentExecutor
from langchain.memory import ConversationBufferWindowMemory

from src.agent import generate_response_from_agent, create_agent
from src.llm import create_chat_llm
from utils import write_message

# tag::setup[]
# Page Config
st.set_page_config("Ebert", page_icon=":movie_camera:")

SYSTEM_MESSAGE = """
You are an expert in knowledge about film and able to provide information about
various movies. Be as helpful as possible for a user and  return as much 
information as possible. Do not answer any questions that do not pertain to
movies, actors, and directors. When in doubt, err on the side of providing
accurate information only.

Please answer all questions with information provided as context and do not
include knowledge derived from your pre-training.
"""


# tag::submit[]
# Submit handler
def handle_submit(user_message: str, agent: AgentExecutor) -> None:
    """
    Handle a submission from a user.

    :param user_message: The message the user provides to the Streamlit app
    :type user_message: str
    :param agent: An agent executor to provide a response to
        :param:`user_message`
    :type agent: AgentExecutor
    """

    with st.spinner('Thinking...'):
        response = generate_response_from_agent(
            prompt=user_message, agent=agent, extraction_key="output",
        )
        sleep(1)
        write_message(role="assistant", content=response, save=True)


# end::submit[]


# tag::session[]
def initialize_session_state() -> None:
    """Initialize the session state for the application."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": " ".join(
                    [
                        "Hi, I'm the GraphAcademy Chatbot!",
                        "How can I help you?",
                    ]
                )
            },
        ]


# end::session[]


# Instantiate the large language model, memory context, and agent.
llm_chatbot = create_chat_llm(
    api_key=st.secrets.open_ai_settings["OPENAI_API_KEY"],
    model_name=st.secrets.open_ai_settings["OPENAI_CHAT_MODEL"],
)
memory = ConversationBufferWindowMemory(
    memory_key="chat_history", k=5, return_messages=True,
)
chat_agent = create_agent(
    tools=[],
    llm=llm_chatbot,
    memory=memory,
    agent_type=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    system_message=SYSTEM_MESSAGE,
)
# end::setup[]


# tag::chat[]
with st.container():
    initialize_session_state()
    # Display messages in Session State
    for message in st.session_state.messages:
        write_message(
            role=message["role"], content=message["content"], save=False,
        )

    # Handle any user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        write_message(role="user", content=prompt, save=True)

        # Generate a response
        handle_submit(user_message=prompt, agent=chat_agent)
# end::chat[]
