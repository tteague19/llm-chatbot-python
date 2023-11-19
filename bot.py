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

# Instantiate the large language model, memory context, and agent.
llm_chatbot = create_chat_llm(
    api_key=st.secrets.open_ai_settings["OPENAI_API_KEY"],
    model_name=st.secrets.open_ai_settings["OPENAI_CHAT_MODEL"]
)
memory = ConversationBufferWindowMemory(
    memory_key="chat_history", k=5, return_messages=True,
)
chat_agent = create_agent(
    tools=[],
    llm=llm_chatbot,
    memory=memory,
    agent_type=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
)
# end::setup[]

# tag::session[]
# Set up Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi, I'm the GraphAcademy Chatbot!  How can I help you?"
        },
    ]


# end::session[]


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


# tag::chat[]
with st.container():
    # Display messages in Session State
    for message in st.session_state.messages:
        write_message(message['role'], message['content'], save=False)

    # Handle any user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        write_message('user', prompt)

        # Generate a response
        handle_submit(user_message=prompt, agent=chat_agent)
# end::chat[]
