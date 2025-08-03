from langchain_ollama import ChatOllama
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tools.calculator import calculator
from langchain.agents import AgentExecutor, create_tool_calling_agent
from memory.vector_store import VectorStore
from tools.document import document_loader, split_document_content
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.tools import tool
from agents.rag_agent import RagAgent
from agents.base_agent import BaseAgent
import os

class SupervisorAgent:
    def __init__(self):
        self.llm = ChatOllama(model='llama3.2:3b', temperature=0.1, max_tokens=1000)
        self.vector_store = VectorStore()
        
        ## initialize the agents
        self.rag_agent = RagAgent()
        self.base_agent = BaseAgent()

        ## create the supervisor tools
        self.tools = self._create_supervisor_tools()

        ## create the supervisor agent
        self.agent = self._create_agent()
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)

    def _create_supervisor_tools(self):
        """Create tools specific to the supervisor agent"""
        
        @tool
        def call_calculator_agent(query):
            """ Call the calculator agent with a query."""
            try:
                response = self.base_agent.run(query)
                return response.get('output', str(response))
            except Exception as e:
                return f"Error calling calculator agent: {e}"
        @tool    
        def call_rag_agent(query):
            """ Call the RAG agent with a query."""
            try:
                response = self.rag_agent.run(query)
                return response.get('output', str(response))
            except Exception as e:
                return f"Error calling RAG agent: {e}"
            
        return [call_calculator_agent, call_rag_agent]
    
    def _create_agent(self):
        system_prompt = SystemMessagePromptTemplate.from_template("""You are a supervisor agent that can delegate tasks to other agents.
        Your role is to:
    1. Analyze the user's request
    2. Decide which specialized agent should handle it
    3. Route the request to the appropriate agent
    4. Return the result to the user

    Available agents:
    - Calculator Agent: Use for math, calculations, arithmetic problems
    - RAG Agent: Use for document-based questions, adding documents to knowledge base, research queries

    Decision guidelines:
    - For math problems (like "what is 5+5", "calculate 15% of 200") → use Calculator Agent
    - For document questions (like "what does the document say about X") → use RAG Agent  
    - For adding documents (like "add this file to knowledge base") → use RAG Agent
    - For general knowledge that might be in documents → use RAG Agent                                                         
    - If unsure, route to RAG Agent for document-based queries                                       
    - If the request is not clear, ask for clarification                                                              
                                                                  """)
        human_prompt = HumanMessagePromptTemplate.from_template("{input}")
        prompt = ChatPromptTemplate.from_messages(
            [system_prompt, human_prompt, ("placeholder", "{agent_scratchpad}")],
        )
        
        return create_tool_calling_agent(
            llm=self.llm,
            prompt=prompt,
            tools=self.tools
        )
    
    def run(self, input_text):
        try:
            response = self.agent_executor.invoke({"input": input_text})
            return response
        except Exception as e:
            return f"An error occurred: {e}"
