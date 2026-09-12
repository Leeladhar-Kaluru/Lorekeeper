from langgraph.graph import StateGraph, START, END

from app.graph.state import AgentState
from app.graph.nodes import GraphNodes

from app.llm.groq_service import HelperService
from app.llm.answer_generator import AnswerGenerator

from app.query.query_engine import QueryEngine

from app.rag.rag_service import RAGService
from app.rag.embeddings import EmbeddingService
from app.rag.vector_store import VectorStore


# ============================================================
# APPLICATION SERVICES
# ============================================================

# Create the HelperService once.
#
# This service is responsible for:
#   1. Understanding the user's question.
#   2. Generating SQL from the QueryPlan.
#
# We reuse the same instance throughout the graph.
helper_service = HelperService()


# QueryEngine uses the same HelperService.
query_engine = QueryEngine(
    helper_service=helper_service
)


# Create the embedding service once.
#
# The VectorStore uses this to convert queries into embeddings
# for semantic similarity search.
embedding_service = EmbeddingService()


# Create the Chroma vector store once.
#
# "discord-lore" is the Chroma collection where our Discord
# message embeddings are stored.
vector_store = VectorStore(
    embedding_function=embedding_service,
    collection_name="discord-lore"
)


# RAGService uses the existing vector store.
#
# RAGService performs:
#   Chroma similarity search
#   +
#   PostgreSQL lookup
#   +
#   surrounding-message expansion
rag_service = RAGService(
    vector_store=vector_store
)


# Create the final answer generator once.
answer_generator = AnswerGenerator()


# ============================================================
# GRAPH NODES
# ============================================================

# Give all application services to GraphNodes.
#
# GraphNodes contains the actual operations that the graph
# can execute.
nodes = GraphNodes(
    helper_service=helper_service,
    query_engine=query_engine,
    rag_service=rag_service,
    answer_generator=answer_generator,
)


# ============================================================
# GRAPH DEFINITION
# ============================================================

# Create a StateGraph using our shared AgentState.
builder = StateGraph(AgentState)


# ------------------------------------------------------------
# REGISTER NODES
# ------------------------------------------------------------

builder.add_node(
    "understand_query",
    nodes.understand_query
)

builder.add_node(
    "set_retrieval_mode",
    nodes.set_retrieval_mode
)

builder.add_node(
    "retrieve_database",
    nodes.retrieve_database
)

builder.add_node(
    "retrieve_semantic",
    nodes.retrieve_semantic
)

builder.add_node(
    "generate_final_context",
    nodes.generate_final_context
)

builder.add_node(
    "generate_answer",
    nodes.generate_answer
)


# ============================================================
# GRAPH FLOW
# ============================================================

# START → understand_query
#
# Every request begins by understanding the user's question.
builder.add_edge(
    START,
    "understand_query"
)

# After we understand the query we need to set the retrieval mode.
# This is done by the set_retrieval_mode node.
builder.add_edge(
    "understand_query",
    "set_retrieval_mode"
)

# ============================================================
# CONDITIONAL ROUTING
# ============================================================
# After we set the retrieval mode we need to route the query to the
# appropriate retrieval node.
#
# nodes.route_retrieval returns one of:
#
#   "database_only"
#   "semantic_only"
#   "both"
#
# LangGraph then uses the mapping below to determine
# which node should execute next.

builder.add_conditional_edges(
    "set_retrieval_mode",
    nodes.route_retrieval,
    {
        "database_only": "retrieve_database",
        "semantic_only": "retrieve_semantic",
        "both": "retrieve_database",
    }
)


# ============================================================
# DATABASE ROUTING
# ============================================================

# If the router selected "both", we need to perform semantic
# retrieval after database retrieval.
#
# If the router selected "database_only", we skip semantic
# retrieval and go directly to context generation.
def after_database(state: AgentState) -> str:

    if state["retrieval_mode"] == "both":
        return "retrieve_semantic"

    return "generate_final_context"


builder.add_conditional_edges(
    "retrieve_database",
    after_database,
)


# ============================================================
# SEMANTIC RETRIEVAL
# ============================================================

# Semantic retrieval always goes to final context generation.
#
# This covers:
#
#   semantic_only
#
# and:
#
#   both
#
# because "both" reaches semantic retrieval after the
# database retrieval node.
builder.add_edge(
    "retrieve_semantic",
    "generate_final_context"
)


# ============================================================
# FINAL CONTEXT
# ============================================================

builder.add_edge(
    "generate_final_context",
    "generate_answer"
)


# ============================================================
# FINAL ANSWER
# ============================================================

builder.add_edge(
    "generate_answer",
    END
)


# ============================================================
# COMPILE GRAPH
# ============================================================

# Compile the StateGraph into an executable graph.
graph = builder.compile()