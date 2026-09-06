from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings

class EmbeddingService(Embeddings):

    def __init__(self,model_name:str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self,texts:list[str]) -> list[list[float]]:

        if not texts:
            raise ValueError("Texts cannot be empty")

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        return embeddings.tolist()

    def embed_query(self,query:str)-> list[float]:

        if not query:
            raise ValueError("Query cannot be empty")

        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        
        return embedding.tolist()