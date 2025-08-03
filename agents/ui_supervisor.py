from langchain_ollama import ChatOllama
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate,MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from agents.base_agent import BaseAgent
from agents.rag_agent import RagAgent
from memory.memory_manager import MemoryManager
import os

class UISupervisor:
    def __init__(self,model_name="llama3.2:3b" ):
        self.llm = ChatOllama(model=model_name, temperature=0.1, max_tokens=1000)
        
        ## initialize the memory manager and agents
        self.memory_manager = MemoryManager(model_name=model_name)
        self.base_agent = BaseAgent(model_name=model_name)
        self.rag_agent = RagAgent()

        ## create supervisor tools
        self.tools = self._create_supervisor_tools()

        ## create the supervisor agent
        self.agent = self._create_agent()
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=False)
        
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
            
        @tool
        def save_to_memory(query):
            """ Save a response to memory."""
            try:
                response = self.memory_manager.add_to_long_term_memory(query)
                return f"Saved to long term memory: {response}"
            except Exception as e:
                return f"Error saving to memory: {e}"
            
        @tool
        def retrieve_from_memory(query):
            """ Retrieve a response from memory."""
            try:
                response = self.memory_manager.get_relevant_history(query)
                if not response:
                    return "No relevant memory found."
                return response
            except Exception as e:
                return f"Error retrieving from memory: {e}"
        return [call_calculator_agent, call_rag_agent, save_to_memory, retrieve_from_memory]

    def _create_agent(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI Assistant that answers the query asked by user and if needed you can use the specialized agents to answer the questions.

             
IMPORTANT RULES:
1. For simple conversational responses like "no", "yes", "hello", "thanks" - respond directly, DON'T use any tools
2. For basic questions that don't require calculation or document lookup - respond directly
3. ONLY use tools when specifically needed:
   - Use calculator_agent ONLY for math problems with numbers to calculate
   - Use rag_agent ONLY for questions about documents or adding documents
             
Use CALCULATOR AGENT only for:
   - Complex calculations: "calculate 2847 * 392 + 1583"
   - Multi-step math problems
Use RAG AGENT only for:
   - "Add this document to knowledge base"
   - "What does the uploaded document say about X"
   - "Search my documents for Y"
   - Questions specifically about USER-UPLOADED documents

Instructions:
- Be helpful and concise
- Route tasks to the appropriate agent
- Use memory to provide personalized assistance
- Always provide clear, actionable responses
             
IMPORTANT: While giving reponse to the user use the first person perspective like "I will calculate this for you" and don't give text in response which you thinks on your own but doesn't want user to see.            
             """),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])
        return create_tool_calling_agent(
            llm=self.llm,
            prompt=prompt,
            tools=self.tools
        )
    def run(self, input_text):
        try:
            ## get conversation history
            chat_history = self.memory_manager.get_conversation_history()
            ## run with memory context
            response = self.agent_executor.invoke({"input": input_text, "chat_history": chat_history.get("chat_history", [])})
            ## save the response to memory
            output = response.get('output', str(response))
            self.memory_manager.add_to_conversation(input_text, output)

            return output
        except Exception as e:
            print(f"Error running UiSupervisor: {e}")
            return str(e)
        
    def get_session_info(self):
        return self.memory_manager.get_session_metadata()
