from langchain_core.documents import Document

from app.rag.vector_store import VectorStore
from app.llm.groq_service import GroqService

class RAGService:

    def __init__(
        self,
        vector_store: VectorStore,
        llm_service: GroqService,
    ):

        self.vector_store = vector_store
        self.llm_service = llm_service

    def answer(
        self,
        query:str,
        k:int = 3
    )->str:

        #let's find relevant documents
        documents = self.vector_store.similarity_search(query=query,k=k)

        #let's build the context from the retrieved documents
        context = self._build_context(documents)

        #let's build the prompt by giving the user query along with the context
        prompt = self._build_prompt(query=query,context=context)

        #let's ask the LLM to answer the question based on the context
        answer = self.llm_service.generate(prompt)

        return answer


    
    def _build_context(self,documents: list[Document])->str:

        return "\n\n".join(
            document.page_content 
            for document in documents
        )
        
    def _build_prompt(self,query:str,context:str)->str:

        return f"""
        You are LoreKeeper, a helpful assistant that answers questions
        using the provided context.

        Use only the information contained in the context.

        If the answer cannot be found in the context, say:
        "I don't know based on the available information."

        Context:
        {context}

        Question:
        {query}
        """
        