from app.config import GEMINI_API_KEY

from langchain_google_genai import GoogleGenerativeAIEmbeddings

class EmbeddingService:
    
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            api_key=GEMINI_API_KEY,
            model="gemini-embedding-001"
        )

    def embed_documents(self,texts: list[str])->list[list[float]]:

        return self.embeddings.embed_documents(texts)

    def embed_query(self,query:str)->list[float]:

        return self.embeddings.embed_query(query)