from dotenv import load_dotenv

from app.llm.groq_service import GroqService
from app.rag.embeddings import EmbeddingService
from app.rag.loaders.pdf_loader import PDFDocumentLoader
from app.rag.rag_service import RAGService
from app.rag.splitter import DocumentSplitter
from app.rag.vector_store import VectorStore

load_dotenv()

def main():
    # 1. Load PDF
    loader = PDFDocumentLoader("data/sample.pdf")
    documents = loader.load()

    print(f"Loaded {len(documents)} pages")

    # 2. Split documents into chunks
    splitter = DocumentSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )

    chunks = splitter.split(documents)

    print(f"Created {len(chunks)} chunks")

    # 3. Create embedding service
    embedding_service = EmbeddingService()

    # 4. Create vector store
    vector_store = VectorStore(
        embedding_function=embedding_service,
    )

    # 5. Store chunks
    vector_store.add_documents(chunks)

    print("Documents stored in Chroma")

    # 6. Create LLM service
    llm = GroqService()

    # 7. Create RAG service
    rag = RAGService(
        vector_store=vector_store,
        llm_service=llm,
    )

    # 8. Ask a question
    query = "Which attributes are important for assessing credit risk?"

    answer = rag.answer(
        query=query,
        k=3,
    )

    print(f"\nQuestion: {query}")
    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
    main()