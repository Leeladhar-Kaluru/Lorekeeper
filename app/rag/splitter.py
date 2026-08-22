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

    def split(self,text:list[Document])->list[Document]:

        if not text:
            raise ValueError("Text cannot be empty")
        
        return self.splitter.split_documents(text)
