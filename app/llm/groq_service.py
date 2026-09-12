import os

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq

from app.query.query_plan import QueryPlan, GeneratedQuery

load_dotenv()


class HelperService:
    """
    Helper LLM service used for structured reasoning tasks.

    Responsibilities:
    - Understand natural-language queries
    - Generate PostgreSQL queries
    """

    def __init__(
        self,
        model_name: str = "openai/gpt-oss-20b",
    ):
        self.llm = ChatGroq(
            model_name=model_name,
            temperature=0,
            max_tokens=4096,
            reasoning_effort="low",
            api_key=os.getenv("HELPER_API_KEY"),
        )

    def understand_query(self, query: str) -> QueryPlan:
        """
        Convert a natural-language question into a QueryPlan.
        """

        prompt = f"""
        You are a query understanding system for a Discord memory application.

        Your task is to analyze the user's question and determine
        what information needs to be retrieved from the database.

        Do NOT answer the user's question.

        Extract only the information required for retrieval.

        The database contains Discord messages.

        Identify:

        - intent
        - author_name
        - channel_name
        - time_range
        - important keywords
        - aggregation requirement

        IMPORTANT:

        1. author_name must contain the exact Discord/database author name
        when a person is mentioned.

        2. channel_name must contain the exact Discord/database channel name
        when a channel is mentioned.

        3. Do not invent authors or channels.

        4. If the user does not mention an author, set author_name to null.

        5. If the user does not mention a channel, set channel_name to null.

        6. If the user does not specify a time range, set time_range to null.

        7. Preserve important semantic keywords.

        8. Identify aggregation requirements such as:
        - most active
        - least active
        - count
        - number
        - average
        - total
        - ranking
        - frequency

        9. Do not generate SQL.

        10. Do not answer the question.

        11. make sure to explain the intent as simple as possible(not some one line sentence)
        such that the query plan will make proper sense and can be used in easy retrieval of
        data from database.

        USER QUESTION
        =============

        {query}
        """

        structured_llm = self.llm.with_structured_output(QueryPlan)

        return structured_llm.invoke(
            [SystemMessage(content=prompt)]
        )

    def generate_query(self, plan: QueryPlan) -> GeneratedQuery:
        """
        Convert a QueryPlan into one safe PostgreSQL SELECT query.
        """

        prompt = f"""
        You are a PostgreSQL query-generation engine for a Discord
        conversation-memory application.

        Your ONLY task is to convert the provided QueryPlan into
        ONE safe, read-only PostgreSQL SELECT query.

        You are NOT responsible for answering the user's question.

        You are responsible only for determining what database
        information must be retrieved.

        When retrieving multiple messages or rows, NEVER return 
        the entire table or an unbounded result set. Always limit 
        the number of returned rows to a reasonable amount that is
        sufficient to answer the user's question. Prefer `LIMIT 20` 
        for general retrieval unless the QueryPlan explicitly requires 
        a different result size. Aggregation queries may return fewer 
        rows naturally. Never use an unbounded `SELECT` when retrieving 
        message records.


        ============================================================
        DATABASE SCHEMA
        ============================================================

        Table: messages

        Columns:

        - message_id
        - server_id
        - server_name
        - channel_id
        - channel_name
        - author_id
        - author_name
        - content
        - timestamp

        ============================================================
        QUERY GENERATION RULES
        ============================================================

        1. Generate exactly ONE SQL query.

        2. The query MUST be a SELECT statement.

        3. NEVER generate:
        - INSERT
        - UPDATE
        - DELETE
        - DROP
        - ALTER
        - CREATE
        - TRUNCATE
        - GRANT
        - REVOKE
        - MERGE
        - EXECUTE
        - CALL
        - or any other modifying operation.

        4. Query ONLY the messages table.

        5. Use ONLY columns listed in the schema.

        6. Never invent tables or columns.

        7. Never modify the database.

        8. Generate exactly one SQL statement.

        9. Do not use SQL comments.

        10. Do not include markdown code fences.

        11. Do not include explanations.

        12. Return only the SQL query.

        13. Use valid PostgreSQL syntax.

        14. Prefer simple and deterministic queries.

        15. Use the QueryPlan as the source of truth.

        16. Do not invent filters that are not present in the QueryPlan.

        17. When author_name is present, filter using the author_name column.

        18. When channel_name is present, filter using the channel_name column.

        19. When time_range is present, translate it into an appropriate
            timestamp condition.

        20. When aggregation is requested, perform the aggregation
            directly in PostgreSQL.

        21. For message retrieval, return useful columns such as:

            message_id
            author_name
            channel_name
            content
            timestamp

        22. Do not answer the question yourself.

        23. The output must be executable PostgreSQL SQL.

        The examples above are guidelines only.
        Do not blindly hardcode them.
        Determine the appropriate query from the QueryPlan.

        ============================================================
        QUERY PLAN
        ============================================================

        {plan.model_dump_json(indent=2)}

        ============================================================
        OUTPUT
        ============================================================

        Return exactly one valid PostgreSQL SELECT query.
        Return nothing except the SQL query.
        """

        structured_llm = self.llm.with_structured_output(
            GeneratedQuery
        )

        return structured_llm.invoke(
            [SystemMessage(content=prompt)]
        )