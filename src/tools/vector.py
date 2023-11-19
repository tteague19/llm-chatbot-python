"""Implements a Neo4j Vector Search Index."""
from typing import Optional

from langchain.schema.embeddings import Embeddings
from langchain.vectorstores.neo4j_vector import Neo4jVector


def create_neo4j_vector_from_existing_index(
        embedding: Embeddings,
        url: str,
        user_name: str,
        password: str,
        index_name: str,
        node_label: str,
        text_node_property: str,
        embedding_node_property: str,
        retrieval_query: Optional[str] = None,
) -> Neo4jVector:
    """
    Create a Neo4jVector instance from an existing index.

    :param embedding: An object that will embed the user input
    :type embedding: Embeddings
    :param url: The URL of a Neo4j database
    :type url: str
    :param user_name: The username associated with the database at
        :param:`url`
    :type user_name: str
    :param password: The password associated with the database at :param:`url`
        and :param:`user_name`
    :type password: str
    :param index_name: The name of the index in the database at :param:`url`
    :type index_name: str
    :param node_label: The label of the nodes used to populate the existing
        index
    :type node_label: str
    :param text_node_property: The name of the property used to create the
        vector embeddings in the existing index
    :type text_node_property: str
    :param embedding_node_property: The name of the property on the nodes in
        the graph database at :param:`url` that contains the vector embedding
    :type embedding_node_property: str
    :param retrieval_query: A Cypher query that allows the user to retrieve
        specific information returned, loaded into documents, and passed to an
        LLM which should also return a text value and a map of metadata,
        defaults to None
    :type retrieval_query: Optional[str]
    :return: A Neo4jVector instance with the specified parameters
    :rtype: Neo4jVector
    """
    return Neo4jVector.from_existing_index(
        embedding=embedding,
        url=url,
        username=user_name,
        password=password,
        index_name=index_name,
        node_label=node_label,
        text_node_property=text_node_property,
        embedding_node_property=embedding_node_property,
        retrieval_query=retrieval_query,
    )
