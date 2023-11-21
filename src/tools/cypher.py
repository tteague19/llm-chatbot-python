"""Implements a Graph Cypher QA Chain for use by the Streamlit front end."""
from langchain.chains import GraphCypherQAChain
from langchain.graphs import Neo4jGraph
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import BasePromptTemplate
from langchain.schema.language_model import BaseLanguageModel


CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer translating user questions into Cypher to 
answer questions about movies and provide recommendations.
Convert the user's question based on the schema.

Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

Fine Tuning:

For movie titles that begin with "The", move "the" to the end. For example 
"The 39 Steps" becomes "39 Steps, The" or "the matrix" becomes "Matrix, The".

Example Cypher Statements:

1. How to find how many degrees of separation there are between two people:
```
MATCH path = shortestPath(
  (p1:Person {{name: "Actor 1"}})-[:ACTED_IN|DIRECTED*]-(p2:Person {{name: "Actor 2"}})
)
WITH path, p1, p2, relationships(path) AS rels
RETURN
  p1 {{ .name, .born, link:'https://www.themoviedb.org/person/'+ p1.tmdbId }} AS start,
  p2 {{ .name, .born, link:'https://www.themoviedb.org/person/'+ p2.tmdbId }} AS end,
  reduce(output = '', i in range(0, length(path)-1) |
    output + CASE
      WHEN i = 0 THEN
       startNode(rels[i]).name + CASE WHEN type(rels[i]) = 'ACTED_IN' THEN ' played '+ rels[i].role +' in 'ELSE ' directed ' END + endNode(rels[i]).title
       ELSE
         ' with '+ startNode(rels[i]).name + ', who '+ CASE WHEN type(rels[i]) = 'ACTED_IN' THEN 'played '+ rels[i].role +' in '
    ELSE 'directed '
      END + endNode(rels[i]).title
      END
  ) AS pathBetweenPeople
```

Schema:
{schema}

Question:
{question}
"""


def create_cypher_qa_chain(
        llm: BaseLanguageModel,
        graph: Neo4jGraph,
        verbose: bool,
        cypher_prompt: BasePromptTemplate,
) -> GraphCypherQAChain:
    """
    Create a Graph Cypher QA Chain from an existing LLM and Neo4j graph.

    :param llm: A large language model (LLM) to use to construct the chain
    :type llm: BaseLanguageModel
    :param graph: A reference to the Neo4j graph to use to construct the chain
    :type graph: Neo4jGraph
    :param verbose: An indicator of whether to provide verbose output from the
        chain
    :type verbose: bool
    :param cypher_prompt: A prompt template for the chain to use when
        generating cypher
    :type cypher_prompt: PromptTemplate
    :return: A Graph Cypher QA Chain for use as a tool
    :rtype: GraphCypherQAChain
    """
    return GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=verbose,
        cypher_prompt=cypher_prompt,
    )
