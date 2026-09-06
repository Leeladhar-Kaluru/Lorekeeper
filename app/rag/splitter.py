from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class DocumentSplitter:

    # splits documnet into small chunks

    def __init__(self,chunk_size:int=200,chunk_overlap:int=40):

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def split(self,documents:list[Document])->list[Document]:

        if not documents:
            raise ValueError("Documents cannot be empty")
        
        chunks = self.splitter.split_documents(documents)

        for index,chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = index

        return chunks
        
