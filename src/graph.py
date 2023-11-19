"""Implements an object to allow interaction between Neo4j and LangChain."""
from langchain.graphs import Neo4jGraph


def create_graph(url: str, user_name: str, password: str) -> Neo4jGraph:
    """
    Create an instance of a graph object for use with LangChain.

    :param url: The URL of a Neo4j graph database
    :type url: str
    :param user_name: The username associated with the graph database at
        :param:`url`
    :type user_name: str
    :param password: The password associated with the graph database at
        :param:`url` and username :param:`user_name`
    :type password: str
    :return: A graph object
    :rtype: Neo4jGraph
    """
    return Neo4jGraph(url=url, username=user_name, password=password)
