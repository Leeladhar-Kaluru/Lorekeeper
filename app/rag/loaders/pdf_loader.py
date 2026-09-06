import pymupdf
from langchain_core.documents import Document

class PDFDocumentLoader:

    def __init__(self,file_path):
        self.file_path = file_path

    def load(self)->list[Document]:
        documents = []
        pdf = pymupdf.open(self.file_path)
        try:
            for page_number, page in enumerate(pdf):
                text = page.get_text()
                if not text.strip():
                    continue
                page_document = Document(
                    page_content = text,
                    metadata = {
                        "source": self.file_path,
                        "page": page_number + 1,
                    }
                )
                documents.append(page_document)
        
        except Exception as e:
            raise e

        finally:
            pdf.close()

        return documents

        
