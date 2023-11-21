# tag::llm[]
import streamlit as st
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(
    openai_api_key=st.secrets.open_ai_settings["OPENAI_API_KEY"],
    model=st.secrets.open_ai_settings["OPENAI_CHAT_MODEL"],
)
# end::llm[]

# tag::embedding[]
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    openai_api_key=st.secrets.open_ai_settings["OPENAI_API_KEY"]
)
# end::embedding[]
