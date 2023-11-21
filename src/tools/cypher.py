"""Implements a Graph Cypher QA Chain for use by the Streamlit front end."""
from langchain.chains import GraphCypherQAChain
from langchain.graphs import Neo4jGraph
from langchain.schema.language_model import BaseLanguageModel


def create_cypher_qa_chain(
        llm: BaseLanguageModel, graph: Neo4jGraph,
) -> GraphCypherQAChain:
    """
    Create a Graph Cypher QA Chain from an existing LLM and Neo4j graph.

    :param llm: A large language model (LLM) to use to construct the chain
    :type llm: BaseLanguageModel
    :param graph: A reference to the Neo4j graph to use to construct the chain
    :type graph: Neo4jGraph
    :return: A Graph Cypher QA Chain for use as a tool
    :rtype: GraphCypherQAChain
    """
    return GraphCypherQAChain.from_llm(llm=llm, graph=graph)
