from app.llm.groq_service import HelperService
from app.query.query_plan import QueryPlan


class QueryEngine:
    """
    Converts a QueryPlan into a PostgreSQL query.
    """

    def __init__(self, helper_service: HelperService):
        # Reuse the HelperService instance created by the application.
        self.helper = helper_service

    def query_generator(self, plan: QueryPlan) -> str:

        # ---------------------------------------------------------
        # Generate SQL from the QueryPlan.
        #
        # understand_query() is NOT called here because the
        # LangGraph understand_query node has already produced
        # the QueryPlan and stored it in AgentState.
        # ---------------------------------------------------------

        generated_query = self.helper.generate_query(plan)

        sql_query = generated_query.sql.strip()

        print("\n========== GENERATED SQL ==========")
        print(sql_query)

        return sql_query