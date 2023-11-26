from src.data.VectorStoreManager import VectorStoreManager

from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.agents import Tool


class DocumentSearch:
    
    def __init__(self, documents: VectorStoreManager, llm = None):
        self.documents = documents
        if llm is None:
            self.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
        else:
            self.llm = llm
        self.retriever = RetrievalQA.from_chain_type(self.llm, "stuff", retriever=documents.vectorStore.as_retriever())


    def asTool(self, topic: str) -> Tool:
        """
        @param topic: Tells the LLM what do the documents pertain to. The exact format is as follows:
        'useful for when you need to answer questions about {topic}. Input should be a fully formed question.'
        """
        return Tool(
            name="Documents QA System",
            func=self.retriever.run,
            description=f"useful for when you need to answer questions about {topic}. Input should be a fully formed question.",
        )
    