"""Implements the Streamlit frontend for the chatbot."""
from time import sleep

import streamlit as st
from langchain.agents import AgentType, AgentExecutor
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferWindowMemory

from src.agent import generate_response_from_agent, create_agent
from src.llm import create_chat_llm, create_embedding_model
from src.tools.vector import create_neo4j_vector_from_existing_index
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

chat_agent = create_agent(
    tools=[],
    llm=llm_chatbot,
    memory=memory,
    agent_type=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    system_message=SYSTEM_MESSAGE,
)

retriever = neo4j_vector.as_retriever()
knowledge_graph_qa = RetrievalQA.from_chain_type(
    llm=llm_chatbot, chain_type="stuff", retriever=retriever,
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
