from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool


class ChatBot:
    
    def __init__(self, llm=None):
        if llm is None:
            self.llm = ChatOpenAI(model="gpt-3.5-turbo")
        else:
            self.llm = llm


    def prompt(self, prompt: str) -> str:
        return self.llm([HumanMessage(content=prompt)]).content
    
    def asTool(self) -> Tool:
        """
        @param topic: Tells the LLM what do the documents pertain to. The exact format is as follows:
        'useful for when you need to answer questions about {topic}. Input should be a fully formed question.'
        """
        return Tool(
            name="Chatbot",
            func=self.prompt,
            description="useful for when the user is not requesting any specific information and just wants to chat. Input should be a fully formed sentence.")