import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq


load_dotenv()


class AnswerGenerator:

    def __init__(
        self,
        model_name: str = "openai/gpt-oss-120b",
    ):
        self.llm = ChatGroq(
            model_name=model_name,
            temperature=0,
            max_tokens=4096,
            reasoning_effort="low",
            api_key=os.getenv("GROQ_API_KEY"),
        )

    def generate(
        self,
        question: str,
        context: str,
    ) -> str:

        system_prompt = """
        You are the final answer generator for LoreKeeper,
        a Discord conversational memory system.

        Your task is to answer the user's question using ONLY
        the retrieved evidence provided to you.

        Rules:

        1. Understand the user's question carefully.

        2. Use only the provided evidence.

        3. Never invent facts that are not supported by the evidence.

        4. If the evidence is insufficient, clearly say so.

        5. Synthesize information from multiple pieces of evidence
        when necessary.

        6. Give a clear, natural and concise answer.

        7. If the retrieved data contains counts, rankings,
        dates, names, or other structured results, report them
        accurately.

        8. Do not mention internal implementation details such as
        QueryPlan, QueryEngine, Chroma, PostgreSQL, LangGraph,
        embeddings, or retrieval pipelines unless the user
        explicitly asks about them.

        9. Do not blindly repeat the retrieved context.

        10. Answer in the language/style appropriate to the user's
            question. The server may contain English, Telugu,
            Hindi, or mixed-language conversations.

        11. When conversational context is relevant, use it to
            understand what people meant.

        12. Do not use outside knowledge as if it came from the
            Discord server.

        OUTPUT FORMAT RULES:

        - Return plain Discord-friendly Markdown.
        - Do NOT use Markdown tables.
        - Do NOT use HTML tags such as <br>, <p>, etc.
        - Do NOT include raw URLs for Discord emoji assets.
        - Prefer headings, bullet points, and short paragraphs.
        - Keep formatting simple and readable in Discord.
        """

        user_prompt = f"""
        USER QUESTION
        =============

        {question}


        RETRIEVED EVIDENCE
        ==================

        {context}


        Using ONLY the retrieved evidence above,
        answer the user's question.
        """

        response = self.llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )

        return response.content