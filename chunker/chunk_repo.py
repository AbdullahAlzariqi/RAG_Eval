
from langchain.text_splitter import RecursiveCharacterTextSplitter




class chunk_repo:
    def __init__(self, file_path, chunk_size=528, chunk_overlap=128):
        print(file_path)
        self.text = self._get_text(file_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks = self._chunk()
    
    
    def _chunk(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = text_splitter.split_text(self.text)
        return chunks
    
    def _get_text(self , file_path):
        # Read the content of the file
        with open(file_path, 'r') as file:
            return file.read()

