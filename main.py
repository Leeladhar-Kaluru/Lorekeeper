from app.config import SERVER_HISTORY_FILE, PDF_FILE
from app.rag.loaders.text_loader import TextDocumentLoader
from app.rag.loaders.pdf_loader import PDFDocumentLoader
from app.rag.splitter import DocumentSplitter
from app.rag.embeddings import EmbeddingService

def main():

    loader = PDFDocumentLoader(PDF_FILE)
    documents = loader.load()

    for i,doc in enumerate(documents):
        print(f"Document {i}")
        print(doc)
        print("\n")

    splitter = DocumentSplitter()
    chunks = splitter.split(documents)

    print(f"Created {len(chunks)} chunks")

    for index, chunk in enumerate(chunks, start=1):
        print(f"\n--- Chunk {index} ---")
        print(chunk.page_content)
        print(f"Metadata: {chunk.metadata}")

    embedding_service = EmbeddingService()

    texts = [chunk.page_content for chunk in chunks]

    vectors = embedding_service.embed_documents(texts)

    for i,vector in enumerate(vectors):
        print(f"Embedding {i}")
        print(vector)
        print("\n")

    print(f"Chunks: {len(chunks)}")
    print(f"Vectors: {len(vectors)}")
    print(f"Vector dimensions: {len(vectors[0])}")


if __name__ == "__main__":
    main()

