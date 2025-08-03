from langchain.tools import tool
from langchain_community.document_loaders import PyPDFLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

@tool
def document_loader(file_path):
    """
    Load a document from the specified file path.
    
    Args:
        file_path: Path to the document file.
        
    Returns:
        Loaded document content as a string.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)
        documents = loader.load()

        return "\n".join([doc.page_content for doc in documents])
    except Exception as e:
        return f"Error loading document: {e}"

@tool
def split_document_content(content, chunk_size=1000, chunk_overlap=200):
    """
    Split the document content into smaller chunks.
    
    Args:
        content: The document content to split.
        chunk_size: Size of each chunk.
        chunk_overlap: Overlap between chunks.
        
    Returns:
        List of document chunks.
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return text_splitter.split_text(content)
    except Exception as e:
        return f"Error splitting document content: {e}"