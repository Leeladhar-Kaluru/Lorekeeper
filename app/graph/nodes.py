from app.graph.state import AgentState

from app.llm.groq_service import HelperService
from app.llm.answer_generator import AnswerGenerator

from app.database.message_repository import MessageRepository
from app.database.connection import SessionLocal

from app.rag.rag_service import RAGService
from app.query.query_engine import QueryEngine


class GraphNodes:

    def __init__(
        self,
        helper_service: HelperService,
        query_engine: QueryEngine,
        rag_service: RAGService,
        answer_generator: AnswerGenerator,
    ):
        # Store the application services so all nodes can reuse
        # the same instances during graph execution.
        self.helper_service = helper_service
        self.query_engine = query_engine
        self.rag_service = rag_service
        self.answer_generator = answer_generator

    def understand_query(self, state: AgentState) -> dict:

        # Get the user's original question from shared state.
        question = state["question"]

        # Convert the natural-language question into a structured
        # QueryPlan.
        query_plan = self.helper_service.understand_query(question)

        print("\n========== QUERY PLAN ==========")
        print(query_plan.model_dump_json(indent=2))

        # Add the QueryPlan to the shared state.
        return {
            "query_plan": query_plan
        }

    def set_retrieval_mode(self, state: AgentState) -> dict:

        plan = state["query_plan"]

        if plan.aggregation:
            retrieval_mode = "database_only"

        elif plan.keywords:
            retrieval_mode = "semantic_only"

        else:
            retrieval_mode = "both"

        print("\n========== RETRIEVAL MODE ==========")
        print(retrieval_mode)

        return {
            "retrieval_mode": retrieval_mode
        }

    def route_retrieval(self, state: AgentState) -> str:

        # Read the routing decision that was already stored in state.
        return state["retrieval_mode"]

    def retrieve_database(self, state: AgentState) -> dict:

        # Get the QueryPlan produced by the previous node.
        query_plan = state["query_plan"]

        # Generate SQL directly from the existing QueryPlan.
        #
        # IMPORTANT:
        # query_generator() no longer understands the question.
        # It only generates SQL from the QueryPlan.
        sql_query = self.query_engine.query_generator(query_plan)

        # Create a short-lived database session.
        session = SessionLocal()

        try:
            # The repository requires the SQLAlchemy session.
            repository = MessageRepository(session)

            # Execute the generated SQL query.
            database_results = repository.execute_query(sql_query)

            print("\n========== DATABASE RESULTS ==========")
            print(database_results)

            # Store both the generated SQL and its results
            # in the shared graph state.
            return {
                "sql_query": sql_query,
                "database_results": database_results,
            }

        finally:
            # Always close the database session after the operation.
            session.close()

    def retrieve_semantic(self, state: AgentState) -> dict:

        # Get the original user question.
        question = state["question"]

        # Perform semantic retrieval using the RAG service.
        semantic_context = self.rag_service.retrieve(
            query=question,
            k=10,
        )

        print("\n========== SEMANTIC CONTEXT ==========")
        print(semantic_context)

        # Store the semantic evidence in shared state.
        return {
            "semantic_context": semantic_context,
        }

    def generate_final_context(self, state: AgentState) -> dict:

        # Get database results if the database retrieval node ran.
        database_results = state.get(
            "database_results",
            []
        )

        # Get semantic context if the semantic retrieval node ran.
        semantic_context = state.get(
            "semantic_context",
            ""
        )

        # Convert database results into readable text.
        database_context = "\n\n".join(
            str(result)
            for result in database_results
        )

        # Build the final context using whichever retrieval
        # mechanisms were selected by the router.
        final_context = f"""
        DATABASE RESULTS
        ================
        {database_context}

        SEMANTICALLY RELEVANT CONVERSATION
        ==================================
        {semantic_context}
        """.strip()

        return {
            "final_context": final_context
        }

    def generate_answer(self, state: AgentState) -> dict:

        # Get the original question and combined evidence.
        question = state["question"]
        final_context = state["final_context"]

        # Generate the final answer using the AnswerGenerator.
        answer = self.answer_generator.generate(
            question,
            final_context,
        )

        # Store the final answer in shared state.
        return {
            "answer": answer,
        }