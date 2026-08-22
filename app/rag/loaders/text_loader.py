from pathlib import Path

class TextDocumentLoader:

    def __init__(self,file_path:str):
        self.file_path = file_path

    def load(self)-> str:
        # read the file and return it's contents

        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        return self.file_path.read_text(encoding='utf-8')
            