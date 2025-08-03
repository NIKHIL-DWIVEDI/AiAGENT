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
            """ Call the RAG agent with a query for document related questions."""
            try:
                response = self.rag_agent.run(query)
                return response.get('output', str(response))
            except Exception as e:
                return f"Error calling RAG agent: {e}"
            
        @tool
        def save_to_memory(query):
            """ Save a response to long term memory."""
            try:
                response = self.memory_manager.add_to_long_term_memory(query)
                return f"Saved to long term memory: {response}"
            except Exception as e:
                return f"Error saving to memory: {e}"
            
        @tool
        def retrieve_from_memory(query):
            """ Retrieve a response from long term memory."""
            try:
                response = self.memory_manager.get_relevant_history(query)
                if not response or not response.get("relavant_history"):
                    return "No relevant memory found."
                return f"From my memory: {response.get('relevant_history', 'No information found')}"
            except Exception as e:
                return f"Error retrieving from memory: {e}"
            
        @tool
        def show_conversation_history(query=None):
            """ Show the conversation history from short term memory."""
            try:
                history = self.memory_manager.get_conversation_history(query)
                if not history:
                    return "No conversation history found."
                return f"Conversation history: {history}"
            except Exception as e:
                return f"Error retrieving conversation history: {e}"
            
        return [call_calculator_agent, call_rag_agent, save_to_memory, retrieve_from_memory, show_conversation_history]
    

    def _create_agent(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI Assistant that answers user queries directly using your knowledge. Only use specialized tools when absolutely necessary.

CRITICAL ROUTING RULES:
1. ANSWER DIRECTLY for these types of questions (DO NOT use any tools):
   - General knowledge: "What is the capital of India?", "Who is the PM of Italy?", "What is 2+3?"
   - Greetings: "hello", "how are you", "thanks"
   - Simple conversations: "yes", "no", basic responses
   - Explanations: "explain quantum physics", "what is AI?"
   - Facts about countries, people, science, history, etc.

2. Use CALCULATOR AGENT only for:
   - Complex calculations: "calculate 2847 * 392 + 1583"
   - Multi-step math problems that are difficult to do mentally

3. Use RAG AGENT only for:
   - "Add this document to knowledge base"
   - "What does my uploaded document say about X?"
   - "Search my documents for Y"
   - "Explain the uploaded document"
   - "Summarize my document"
   - Questions specifically about USER-UPLOADED documents

4. Use MEMORY tools for:
   - "What was my first question?" → Use show_conversation_history
   - "What did I ask before?" → Use show_conversation_history
   - "Remember that I like pizza" → Use save_to_memory
   - "What do you remember about me?" → Use retrieve_from_memory

5. Use SHOW_CONVERSATION_HISTORY when:
   - User asks about previous questions/conversations
   - "What was my first question?"
   - "What did we talk about?"

IMPORTANT EXAMPLES:
- "What is 2+3?" → Answer: "2+3 equals 5."
- "Who is PM of Italy?" → Answer: "Giorgia Meloni is the current Prime Minister of Italy."
- "What was my first question?" → Use show_conversation_history tool
- "Explain the uploaded document" → Use call_rag_agent tool

Instructions:
- Be helpful and conversational
- Use first person: "I can help you with that"
- Answer directly using your built-in knowledge for general questions
- Only use tools when explicitly needed for the specific use cases above"""),
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
