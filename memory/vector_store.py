import chromadb
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import os

class VectorStore:
    def __init__(self,collection_name="documents",persist_directory="db"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        ## intilialize ollama embeddings
        self.embeddings = OllamaEmbeddings(model="llama3.2:3b")

        ## create persist directory if it does not exist
        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory)

        ## initialize vector store
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )

    ## add documents to vector store
    def add_documents(self, documents):
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add.
        """
        try:
            self.vector_store.add_texts(texts=documents)
            print(f"Added {len(documents)} documents to the vector store.")
        except Exception as e:
            print(f"Error adding documents: {e}")

    ## search for the documents in the vector store
    def search(self, query, k=5):
        """
        Search for documents in the vector store.
        
        Args:
            query: The search query.
            k: Number of results to return.
        
        Returns:
            List of search results.
        """
        try:
            results = self.vector_store.similarity_search(query, k=k)
            return results
        except Exception as e:
            print(f"Error searching documents: {e}")
            return []