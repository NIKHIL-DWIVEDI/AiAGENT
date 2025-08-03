from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
from agents.base_agent import BaseAgent
from agents.rag_agent import RagAgent
from memory.memory_manager import MemoryManager

class MemorySupervisor:
    def __init__(self,model_name="llama3.2:3b" ):
        self.llm = ChatOllama(model=model_name, temperature=0.1, max_tokens=1000)
        self.memory_manager = MemoryManager()
        self.base_agent = BaseAgent()
        self.rag_agent = RagAgent()

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
                return f"Retrieved from memory: {response}"
            except Exception as e:
                return f"Error retrieving from memory: {e}"
            
        return [call_calculator_agent, call_rag_agent, save_to_memory, retrieve_from_memory]

    def _create_agent(self):
        prompt = ChatPromptTemplate.from_messages(
            [
("system", """You are a Memory-Enhanced Supervisor Agent that coordinates specialized agents and manages memory.

Your capabilities:
1. Route queries to specialized agents (Calculator, RAG)
2. Save important information to long-term memory
3. Recall relevant information from past conversations
4. Maintain conversation context

Available tools:
- call_calculator_agent: For math and calculations
- call_rag_agent: For document-based questions
- save_important_info: Save important facts for later
- recall_information: Retrieve relevant past information

Memory guidelines:
- Save user preferences, important facts, or recurring topics
- Before answering, check if relevant past information exists
- Use conversation history to maintain context"""),
                MessagesPlaceholder(variable_name="conversation_history",optional=True),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ]
        )
        return create_tool_calling_agent(
            llm=self.llm,
            prompt=prompt,
            tools=self.tools
        )

    def run(self, input_text):
        try:
        ## get conversation history
            conversation_history = self.memory_manager.get_conversation_history(query=input_text)

            ## run the agent with conversation history
            response = self.agent_executor.invoke({"input": input_text, "conversation_history": conversation_history})

            ## save the response to memory
            output = response.get('output', str(response))
            self.memory_manager.add_to_conversation(input_text, output)

            return response
        except Exception as e:
            return f"An error occurred: {e}"
        
    def get_memory_stats(self):
        """Get memory statistics"""
        session_info = self.memory_manager.get_session_metadata()
        return {
            "session_id": self.memory_manager.current_session,
            "messages_in_session": session_info.get("messages", 0),
            "session_created": session_info.get("created", "Unknown")
        }