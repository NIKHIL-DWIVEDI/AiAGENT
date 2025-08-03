from datetime import datetime
import json
from langchain.memory import ConversationBufferMemory, VectorStoreRetrieverMemory
from langchain_core.tools import tool
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
import os

class MemoryManager:
    def __init__(self,model_name="llama3.2:3b", persist_directory="db" ):
        self.model_name = model_name
        self.persist_directory = persist_directory

        os.makedirs(self.persist_directory, exist_ok=True)

        ## short term memory
        self.short_term_memory = ConversationBufferMemory(
            human_prefix="User",
            ai_prefix="AI",
            memory_key="history")
        
        ## long term memory
        self.embeddings = OllamaEmbeddings(model=self.model_name)
        self.long_term_memory  = self._setup_long_term_memory()

        ##session metadata
        self.session_file = os.path.join(self.persist_directory, "session_metadata.json")
        self.current_session = self._create_session_metadata()

    ## create vector store based storage for long term memory
    def _setup_long_term_memory(self):
        """Initialize the long term memory using a vector store."""
        try:

            vector_store = Chroma(
                collection_name="long_term_memory",
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )

            ## create retriever memory
            retriever_memory = VectorStoreRetrieverMemory(
                retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
                memory_key="relevant_history",
                return_messages=True
            )

            return retriever_memory
        except Exception as e:
            print(f"Error setting up long term memory: {e}")
            return None
        
    def _create_session_metadata(self):
        """Create metadata for the current session."""
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        session_metadata = {}
        ## load existing session metadata if available
        if os.path.exists(self.session_file):
            try:
                with open(self.session_file, 'r') as f:
                    session_metadata = json.load(f)
            except Exception as e:
                print(f"Error loading session metadata: {e}")
                session_metadata = {}

        ## create new session metadata
        session_metadata[session_id] = {
            "created_at": datetime.now().isoformat(),
            "messages": 0
        }

        ## save the session metadata
        try:
            with open(self.session_file, 'w') as f:
                json.dump(session_metadata, f, indent=4)
        except Exception as e:
            print(f"Error saving session metadata: {e}") 

        return session_id
    
    def get_session_metadata(self):
        """Get metadata for the current session."""
        try:
            with open(self.session_file, 'r') as f:
                all_session_metadata = json.load(f)
            return all_session_metadata.get(self.current_session, {})
        except Exception as e:
            print(f"Error retrieving session metadata: {e}")
            return {}
        
    def _update_session_metadata(self):
        """Update the session metadata with the number of messages."""
        ## load all the session metadata
        try:
            with open(self.session_file, 'r') as f:
                all_session_metadata = json.load(f)
        except Exception as e:
            print(f"Error loading session metadata: {e}")
            return
        
        ## update the current session metadata
        if self.current_session in all_session_metadata:
            all_session_metadata[self.current_session]['messages'] += 1

        try:
            with open(self.session_file, 'w') as f:
                json.dump(all_session_metadata, f, indent=4)
        except Exception as e:
            print(f"Error updating session metadata: {e}")

    def add_to_short_term_memory(self, human_message, ai_message):
        """Add a message to the short term memory."""
        try:
            self.short_term_memory.save_context(
                {"input":human_message}, {"output":ai_message})
            self._update_session_metadata(1)
        except Exception as e:
            print(f"Error adding to short term memory: {e}")

    def add_to_long_term_memory(self, content, metadata=None):
        if self.long_term_memory:
            if metadata is None:
                metadata = {
                    "timestamp": datetime.now().isoformat(),
                    "session_id": self.current_session
                }

            try:
                self.long_term_memory.save_context(
                    {"input": content}, {"output": ""}
                )
            except Exception as e:
                print(f"Error adding to long term memory: {e}")

    def add_to_conversation(self,human_input, ai_response):
        """Add a conversation to the memory."""
        self.short_term_memory.save_context(
            {"input": human_input}, {"output": ai_response}
        )
        self._update_session_metadata()
              

    def get_conversation_history(self,query=None):
        """Get the conversation history from short term memory."""
        try:
            return self.short_term_memory.load_memory_variables({"prompt":query})
        except Exception as e:
            print(f"Error retrieving conversation history: {e}")
            return []
        
    def get_relevant_history(self, query):
        """Get relevant history from long term memory."""
        if self.long_term_memory:
            try:
                return self.long_term_memory.load_memory_variables({"prompt": query})
            except Exception as e:
                print(f"Error retrieving relevant history: {e}")
                return {}
        else:
            print("Long term memory is not initialized.")
            return {}
            


