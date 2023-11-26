# from AgentBase import AgentBase

from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType


class AgentChatGPT():
    def __init__(self, llm = None):
        if llm is None:
            self.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
        else:
            self.llm = llm
        self.agent = None
    
    def init(self, tools: list, verbose: bool = True):
        self.agent = initialize_agent(tools, self.llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=verbose)
    
    def run(self, prompt: str) -> str:
        if self.agent is None:
            raise RuntimeError("Agent has not been initialized. Please call the init method first.")
        return self.agent.run(prompt)
