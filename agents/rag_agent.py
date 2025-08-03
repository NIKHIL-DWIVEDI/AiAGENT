from langchain_ollama import ChatOllama
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tools.calculator import calculator
from langchain.agents import AgentExecutor, create_tool_calling_agent
from memory.vector_store import VectorStore
from tools.document import document_loader,split_document_content
from langchain_community.document_loaders import PyPDFLoader,TextLoader
from langchain_core.tools import tool
import os

class RagAgent:
    def __init__(self):
        self.llm = ChatOllama(model='llama3.2:3b', temperature=0.1, max_tokens=1000)
        self.vector_store = VectorStore()
        
        ## create the RAG specific tools
        self.tools = self._create_rag_tools()

        ## create the agent
        self.agent = self._create_agent()
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)

    def _create_rag_tools(self):
        """Create tools specific to RAG functionality"""
        
        @tool
        def add_document_to_knowledge(file_path: str) -> str:
            """
            Add a document to the knowledge base for future queries.
            
            Args:
                file_path: Path to the document file
                
            Returns:
                Status of the operation
            """
            try:
                # Load document
                if not os.path.exists(file_path):
                    return f"Error: File {file_path} not found"
                                
                if file_path.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                else:
                    loader = TextLoader(file_path)
                
                docs = loader.load()
                
                # Split into chunks
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                
                chunks = []
                metadata = []
                for doc in docs:
                    doc_chunks = splitter.split_text(doc.page_content)
                    chunks.extend(doc_chunks)
                    # Add metadata for each chunk
                    for _ in doc_chunks:
                        metadata.append({"source": file_path})
                
                # Add to vector store
                result = self.vector_store.add_documents(chunks, metadata)
                return f"Successfully processed {file_path}: {result}"
                
            except Exception as e:
                return f"Error processing document: {str(e)}"
        
        @tool
        def search_knowledge_base(query: str, num_results: int = 3) -> str:
            """
            Search the knowledge base for relevant information.
            
            Args:
                query: The search query
                num_results: Number of results to return
                
            Returns:
                Relevant information from the knowledge base
            """
            try:
                results = self.vector_store.search(query, k=num_results)
                
                if not results or isinstance(results, str):
                    return "No relevant information found in knowledge base."
                
                # Format results
                formatted_results = "\n\n--- Relevant Information ---\n"
                for i, result in enumerate(results):
                    formatted_results += f"Result {i+1}:\n{result}\n\n"
                
                return formatted_results
                
            except Exception as e:
                return f"Error searching knowledge base: {str(e)}"
        
        return [add_document_to_knowledge, search_knowledge_base]
    


    def _create_agent(self):
        system_prompt = SystemMessagePromptTemplate.from_template("""You are a helpful RAG assistant." 
        "You can load documents, split them into chunks, and answer questions based on the content of the documents." 
        Your capabilities:
        1. Add documents to your knowledge base using add_document_to_knowledge tool
        2. Search your knowledge base using search_knowledge_base tool
        3. Answer questions based on retrieved information

        When answering questions:
        - First search your knowledge base for relevant information
        - Base your answers on the retrieved information
        - If no relevant information is found, say so clearly                                                          
        """)
        human_prompt = HumanMessagePromptTemplate.from_template("{input}")
        prompt = ChatPromptTemplate.from_messages(
            [system_prompt, human_prompt, ("placeholder", "{agent_scratchpad}")],
        )
        
        ## bind tools to LLM first
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
    





