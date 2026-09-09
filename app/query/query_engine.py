from app.llm.groq_service import HelperService
from app.query.query_plan import QueryPlan


class QueryEngine:
    """
    Converts a natural-language question into
    a PostgreSQL query.
    """

    def __init__(self):
        self.helper = HelperService()

    def query_generator(self, query: str) -> str:

        # ---------------------------------------------------------
        # 1. Understand the user's question
        # ---------------------------------------------------------

        plan: QueryPlan = self.helper.understand_query(query)

        print("\n========== QUERY PLAN ==========")
        print(plan.model_dump_json(indent=2))

        # ---------------------------------------------------------
        # 2. Generate SQL
        # ---------------------------------------------------------

        generated_query = self.helper.generate_query(plan)

        sql_query = generated_query.sql.strip()

        print("\n========== GENERATED SQL ==========")
        print(sql_query)

        return sql_query