from langchain_chroma import Chroma
from langchain_core.documents import Document


class VectorStore:
    """Stores documents and their embeddings in Chroma."""

    def __init__(
        self,
        embedding_function,
        persist_directory: str = "data/chroma",
        collection_name: str = "lorekeeper",
    ):
        self.store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=persist_directory,
        )

    def _create_document_id(self, document: Document) -> str:
        """Create a stable ID for a document chunk."""

        source = document.metadata["source"]
        chunk_index = document.metadata["chunk_index"]

        if source=="discord":
            message_id = document.metadata["message_id"]
            server_id = document.metadata["server_id"]
            return f"{source}:{server_id}:{message_id}:chunk-{chunk_index}"

        page = document.metadata.get("page", 0)
        return f"{source}:page-{page}:chunk-{chunk_index}"

    def add_documents(self, documents: list[Document]) -> None:
        """Add or update documents using stable IDs."""

        ids = [
            self._create_document_id(document)
            for document in documents
        ]

        self.store.add_documents(
            documents=documents,
            ids=ids,
        )

    def similarity_search(
        self,
        query: str,
        k: int = 4,
    ) -> list[Document]:
        """Find documents most similar to a query."""

        return self.store.similarity_search(
            query,
            k=k,
        )