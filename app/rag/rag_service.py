from langchain_core.documents import Document

from app.rag.vector_store import VectorStore

from app.database.connection import SessionLocal
from app.database.message_repository import MessageRepository


class RAGService:

    def __init__(
        self,
        vector_store: VectorStore,
    ):
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        k: int = 25,
    ) -> str:
        """
        Retrieve semantically relevant Discord conversation
        and expand it with nearby messages from PostgreSQL.

        This service ONLY retrieves evidence.
        It does not generate an answer.
        """

        session = SessionLocal()

        try:

            db_service = MessageRepository(session)

            # -----------------------------------------------------
            # 1. Semantic retrieval from Chroma
            # -----------------------------------------------------

            documents = self.vector_store.similarity_search(
                query=query,
                k=k,
            )

            print(f"\nChroma documents: {len(documents)}")

            # -----------------------------------------------------
            # 2. Expand retrieved messages using PostgreSQL
            # -----------------------------------------------------

            expanded_context = self._expand_context(
                documents,
                db_service,
            )

            # -----------------------------------------------------
            # 3. Combine semantic + expanded evidence
            # -----------------------------------------------------

            semantic_context = self._build_context(documents)

            return self._combine_context(
                semantic_context,
                expanded_context,
            )

        finally:
            session.close()

    def _expand_context(
        self,
        documents: list[Document],
        db_service: MessageRepository,
    ) -> str:

        unique_messages = {}

        relevant_message_ids = [
            int(document.metadata["message_id"])
            for document in documents
            if document.metadata.get("message_id") is not None
        ]

        for message_id in relevant_message_ids:

            target_message = db_service.get_message(
                message_id
            )

            if not target_message:
                continue

            nearby_messages = db_service.get_nearby_messages(
                channel_id=target_message.channel_id,
                timestamp=target_message.timestamp,
                before=10,
                after=10,
            )

            for message in nearby_messages:
                unique_messages[message.message_id] = message

        context = sorted(
            unique_messages.values(),
            key=lambda message: message.timestamp,
        )

        print(f"Expanded context: {len(context)} messages")

        return "\n\n".join(
            [
                f"[Channel: #{message.channel_name}]\n"
                f"[Author: {message.author_name}]\n"
                f"[Time: {message.timestamp}]\n"
                f"Message: {message.content}"
                for message in context
            ]
        )

    def _build_context(
        self,
        documents: list[Document],
    ) -> str:

        context = []

        for document in documents:

            metadata = document.metadata

            author = metadata.get(
                "author_name",
                "Unknown",
            )

            channel = metadata.get(
                "channel_name",
                "Unknown",
            )

            timestamp = metadata.get(
                "timestamp",
                "Unknown",
            )

            context.append(
                f"[Channel: #{channel}]\n"
                f"[Author: {author}]\n"
                f"[Time: {timestamp}]\n"
                f"Message: {document.page_content}"
            )

        return "\n\n".join(context)

    def _combine_context(
        self,
        semantic_context: str,
        expanded_context: str,
    ) -> str:

        return f"""
        SEMANTICALLY RELEVANT MESSAGES
        ==============================

        {semantic_context}


        SURROUNDING CONVERSATION
        ========================

        {expanded_context}
        """.strip()