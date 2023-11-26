from typing import Literal

from langchain.document_loaders import DirectoryLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.vectorstores.faiss import FAISS


class VectorStoreManager:

    def __init__(self, type: Literal["Chroma", "FAISS"]):
        if type not in ["Chroma", "FAISS"]:
            raise AttributeError(f"Vector store '{type}' is not supported. Choose either Chroma or FAISS.")
        
        self.type = type
        self.vectorStore = None
        
    
    def load(self, dataPath: str):
        # TODO: Refactor to be database intedependent
        loader = DirectoryLoader(dataPath, glob="*.json", loader_cls=JSONLoader,
                                 loader_kwargs={"jq_schema": ".Pages[].Text"})
        
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'})
        
        if self.type == "Chroma":
            self.vectorStore = Chroma.from_documents(texts, embeddings)
        elif self.type == "FAISS":
            self.vectorStore = FAISS.from_documents(texts, embeddings)
        else:
            raise AttributeError(f"Vector store '{type}' is not supported. Choose either Chroma or FAISS.")
    
    
    def get(self):
        return self.vectorStore