"""Implements the Streamlit frontend for the chatbot."""
from functools import partial
from time import sleep
from typing import Optional, Callable

import streamlit as st
from langchain.agents import AgentType
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import Tool

from src.agent import create_agent, generate_response_from_agent
from src.llm import create_chat_llm, create_embedding_model
from src.tools.vector import create_neo4j_vector_from_existing_index
from utils import write_message

ResponseFunction = Callable[[str], Optional[str]]

# tag::setup[]
# Page Config
st.set_page_config("Ebert", page_icon=":movie_camera:")

SYSTEM_MESSAGE = """
You are a movie expert providing information about movies.
Be as helpful as possible and return as much information as possible.
Do not answer any questions that do not relate to movies, actors or directors.

Do not answer any questions using your pre-trained knowledge, only use the 
information provided in the context.
"""

RETRIEVAL_QUERY = """
RETURN
    node.plot AS text,
    score,
    {
        title: node.title,
        directors: [ (person)-[:DIRECTED]->(node) | person.name ],
        actors: [ (person)-[r:ACTED_IN]->(node) | [person.name, r.role] ],
        tmdbId: node.tmdbId,
        source: 'https://www.themoviedb.org/movie/'+ node.tmdbId
    } AS metadata
"""

VECTOR_SEARCH_TOOL_DESC = """
Provides information about movie plots using Vector Search
"""


# tag::submit[]
# Submit handler
def handle_submit(
        user_message: str,
        response_generation_func: ResponseFunction,
) -> None:
    """
    Handle a submission from a user.

    :param user_message: The message the user provides to the Streamlit app
    :type user_message: str
    :param response_generation_func: A function to pass a prompt and receive
        a response
    :type response_generation_func: ResponseFunction
    """
    with st.spinner('Thinking...'):
        response = response_generation_func(user_message)
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
embeddings = create_embedding_model(
    api_key=st.secrets.open_ai_settings["OPENAI_API_KEY"],
    model_name=st.secrets.open_ai_settings["OPENAI_EMBEDDING_MODEL"],
)
neo4j_vector = create_neo4j_vector_from_existing_index(
    embedding=embeddings,
    url=st.secrets.neo4j_settings["URI"],
    user_name=st.secrets.neo4j_settings["USERNAME"],
    password=st.secrets.neo4j_settings["PASSWORD"],
    index_name=st.secrets.neo4j_settings["INDEX_NAME"],
    node_label=st.secrets.neo4j_settings["NODE_LABEL"],
    text_node_property=st.secrets.neo4j_settings["TEXT_NODE_PROP"],
    embedding_node_property=st.secrets.neo4j_settings["EMBEDDING_NODE_PROP"],
    retrieval_query=RETRIEVAL_QUERY,
)

retriever = neo4j_vector.as_retriever()
knowledge_graph_qa = RetrievalQAWithSourcesChain.from_llm(
    llm=llm_chatbot, retriever=retriever, chain_type="stuff",
)
tools = [
    Tool.from_function(
        name="Vector Search Index",
        description=VECTOR_SEARCH_TOOL_DESC,
        func=knowledge_graph_qa,
    )
]

chat_agent = create_agent(
    tools=tools,
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
        response_func = partial(
            generate_response_from_agent,
            agent=chat_agent,
            extraction_key="output",
        )
        handle_submit(
            user_message=prompt, response_generation_func=response_func,
        )
# end::chat[]
