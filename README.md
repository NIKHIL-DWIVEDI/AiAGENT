# 🤖 AiAGENT - Local Multi-Agent AI System

> **⚠️ Development Status**: This project is currently under active development. Features and documentation will be updated as the project evolves.

A sophisticated multi-agent AI system built with LangChain and Ollama that provides intelligent task routing, persistent memory, and document processing capabilities - all running locally for complete privacy and control.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-green.svg)](https://python.langchain.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.47+-red.svg)](https://streamlit.io/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-orange.svg)](https://ollama.ai/)

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Components](#-components)
- [Configuration](#️-configuration)
- [Development Roadmap](#-development-roadmap)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

AiAGENT is a privacy-focused, locally-run AI system that leverages multiple specialized agents to handle different types of tasks. Built on top of LangChain and powered by Ollama's local LLM inference, it provides a complete AI assistant experience without sending any data to external servers.

### Why AiAGENT?

- **🔒 Complete Privacy**: All processing happens locally on your machine
- **🤖 Multi-Agent Intelligence**: Specialized agents for different tasks
- **🧠 Persistent Memory**: Remembers conversations and learns from interactions
- **📚 Document Understanding**: Process and query your documents with AI
- **💬 Natural Interaction**: Chat-based interface powered by Streamlit
- **🔌 Extensible**: Easy to add new agents and capabilities

---

## 🌟 Features

### Core Capabilities

- **🎯 Multi-Agent Architecture**: Specialized agents work together to handle different tasks
  - **Calculator Agent**: Handles mathematical calculations and arithmetic operations
  - **RAG Agent**: Processes documents and answers questions based on uploaded content
  - **Memory Agent**: Manages conversation history and user preferences

- **🧠 Dual Memory System**: 
  - **Short-term Memory**: Maintains conversation context within a session
  - **Long-term Memory**: Persistent vector-based storage for important information

- **📚 Document Intelligence**: 
  - Upload PDF and text documents
  - Semantic search across your document collection
  - Question-answering based on uploaded content
  - Automatic document chunking and indexing

- **🔄 Smart Query Routing**: 
  - Automatically determines which agent should handle each query
  - Direct answers for general knowledge questions
  - Tool-based routing for specialized tasks

- **💬 Interactive UI**: 
  - Beautiful Streamlit-based chat interface
  - Real-time document upload
  - Session management
  - Conversation history

- **🔒 Privacy-First Design**: 
  - All data stays on your local machine
  - No external API calls (except for Ollama if running remotely)
  - Full control over your data

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                       USER INTERFACE LAYER                       │
├──────────────────────────────────────────────────────────────────┤
│  📱 Streamlit Web App (app.py)                                   │
│  • Chat Interface                                                │
│  • File Upload Widget                                            │
│  • Session State Management                                      │
│  • Sidebar with Agent Info                                       │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATION LAYER                           │
├──────────────────────────────────────────────────────────────────┤
│  🎯 UISupervisor (ui_supervisor.py)                              │
│  • Main coordinator for all UI requests                          │
│  • Intelligent query routing                                     │
│  • Memory integration                                            │
│  • Response coordination                                         │
│  • Conversation flow management                                  │
│                                                                  │
│  Alternative Supervisors (in development):                       │
│  • SupervisorAgent (supervisor_agent.py)                        │
│  • MemorySupervisor (memory_supervisor.py)                      │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│                     SPECIALIZED AGENTS                           │
├──────────────────────────────────────────────────────────────────┤
│  🧮 BaseAgent              📚 RagAgent                            │
│  (base_agent.py)           (rag_agent.py)                        │
│  • Math operations         • Document loading (PDF/TXT)          │
│  • Calculator tool         • Text chunking & splitting           │
│  • Arithmetic queries      • Vector storage                      │
│                            • Semantic search                     │
│                            • Knowledge base queries              │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│                        MEMORY SYSTEM                             │
├──────────────────────────────────────────────────────────────────┤
│  🧠 MemoryManager (memory_manager.py)                            │
│  • Short-term: ConversationBufferMemory                          │
│    - Session-based conversation history                          │
│    - Human/AI message tracking                                   │
│  • Long-term: VectorStoreRetrieverMemory                         │
│    - Persistent semantic memory                                  │
│    - ChromaDB-backed storage                                     │
│  • Session Management                                            │
│    - Metadata tracking                                           │
│    - Session IDs and timestamps                                  │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│                      STORAGE & TOOLS LAYER                       │
├──────────────────────────────────────────────────────────────────┤
│  💾 VectorStore            🛠️ Tools                              │
│  (vector_store.py)         (tools/)                              │
│  • ChromaDB integration    • calculator.py                       │
│  • Ollama embeddings       • document.py                         │
│  • Similarity search       • Document loader                     │
│  • Document persistence    • Text splitter                       │
└──────────────────────────────────────────────────────────────────┘
```

### Architecture Flow

1. **User Input**: User sends a message through the Streamlit chat interface
2. **UI Supervisor**: Receives the input and analyzes the query type
3. **Route Decision**: 
   - Direct answer for general knowledge
   - Calculator Agent for math problems
   - RAG Agent for document-related queries
   - Memory tools for conversation history
4. **Agent Execution**: Selected agent processes the query using its specialized tools
5. **Memory Update**: Conversation is saved to both short-term and long-term memory
6. **Response**: Result is displayed in the chat interface

---

## 🚀 Installation

### Prerequisites

Before installing AiAGENT, ensure you have:

- **Python 3.8+** installed on your system
- **Ollama** installed and running ([Download Ollama](https://ollama.ai/))
- **4GB+ RAM** recommended for smooth operation
- **Disk Space**: At least 2GB for models and dependencies

### Step-by-Step Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/NIKHIL-DWIVEDI/AiAGENT.git
   cd AiAGENT
   ```

2. **Create a Virtual Environment** (Recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install and Setup Ollama**
   
   **On macOS/Linux:**
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ```
   
   **On Windows:**
   Download and install from [ollama.ai](https://ollama.ai/)

5. **Pull the Required Model**
   ```bash
   ollama pull llama3.2:3b
   ```
   
   Note: You can use other models by changing the model name in the configuration. Available models: `llama3.2`, `mistral`, `codellama`, etc.

6. **Verify Installation**
   ```bash
   ollama list  # Should show llama3.2:3b
   ```

---

## 💻 Usage

### Starting the Application

1. **Start Ollama** (if not already running)
   ```bash
   ollama serve
   ```

2. **Launch the Application**
   ```bash
   streamlit run app.py
   ```

3. **Access the Interface**
   - Open your browser and navigate to `http://localhost:8501`
   - The application should load with a chat interface

### Using the Chat Interface

**General Conversations:**
```
You: Hello! What can you do?
AI: I'm a multi-agent AI system that can help you with calculations, 
    document analysis, and remember important information from our conversations.
```

**Mathematical Calculations:**
```
You: What is 2847 * 392 + 1583?
AI: [Uses Calculator Agent] The result is 1,118,907
```

**Document Upload and Querying:**
1. Use the sidebar to upload a document (PDF or TXT)
2. Wait for the success message
3. Ask questions about the document:
```
You: What are the main points in the uploaded document?
AI: [Uses RAG Agent and searches the document] Based on the document, 
    the main points are...
```

**Memory Functions:**
```
You: Remember that I prefer Python for data analysis
AI: [Saves to memory] I've saved that preference!

You: What did I say about my programming preferences?
AI: [Retrieves from memory] You mentioned that you prefer Python for data analysis.

You: What was my first question?
AI: [Shows conversation history] Your first question was "Hello! What can you do?"
```

### Available Commands

- **General queries**: Ask anything and get intelligent responses
- **Math operations**: "Calculate X", "What is X + Y", etc.
- **Document management**: Upload files via sidebar, then ask questions
- **Memory queries**: "What did I ask before?", "What do you remember about me?"
- **Clear chat**: Use the "🗑️ Clear Chat" button in the sidebar

---

## 📁 Project Structure

```
AiAGENT/
├── agents/                    # Agent implementations
│   ├── __init__.py
│   ├── base_agent.py         # Calculator agent for math operations
│   ├── rag_agent.py          # RAG agent for document processing
│   ├── ui_supervisor.py      # Main supervisor for UI (active)
│   ├── supervisor_agent.py   # Alternative supervisor (in development)
│   └── memory_supervisor.py  # Memory-focused supervisor (in development)
│
├── memory/                    # Memory management components
│   ├── __init__.py
│   ├── memory_manager.py     # Short & long-term memory handler
│   └── vector_store.py       # ChromaDB vector storage wrapper
│
├── tools/                     # Reusable tool implementations
│   ├── __init__.py
│   ├── calculator.py         # Mathematical expression evaluator
│   └── document.py           # Document loading and processing utilities
│
├── config/                    # Configuration files (extensible)
│   └── __init__.py
│
├── db/                        # Local database storage (created at runtime)
│   ├── chroma.sqlite3        # ChromaDB database
│   └── session_metadata.json # Session tracking
│
├── uploaded_docs/            # User-uploaded documents (created at runtime)
│
├── app.py                    # Main Streamlit application entry point
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore patterns
└── README.md                # This file
```

### Key Files Explained

- **`app.py`**: Entry point of the application, sets up Streamlit UI and handles user interactions
- **`agents/ui_supervisor.py`**: Main orchestrator that routes queries to appropriate agents
- **`agents/base_agent.py`**: Handles mathematical calculations using LangChain tools
- **`agents/rag_agent.py`**: Manages document upload, indexing, and retrieval
- **`memory/memory_manager.py`**: Implements dual memory system (short-term + long-term)
- **`memory/vector_store.py`**: Wrapper around ChromaDB for vector storage operations
- **`tools/calculator.py`**: Safe mathematical expression evaluator
- **`tools/document.py`**: Document loading and splitting utilities

---

## 🔧 Components

### Agents

#### 1. **UISupervisor** (`ui_supervisor.py`)
The main orchestrator that coordinates all other agents and manages the conversation flow.

**Key Responsibilities:**
- Query analysis and routing
- Memory integration
- Response coordination
- Conversation context management

**Tools:**
- `call_calculator_agent`: Routes math queries
- `call_rag_agent`: Routes document queries
- `save_to_memory`: Saves information to long-term memory
- `retrieve_from_memory`: Retrieves relevant past information
- `show_conversation_history`: Displays past conversations

#### 2. **BaseAgent** (`base_agent.py`)
Specialized agent for mathematical operations and calculations.

**Capabilities:**
- Arithmetic operations
- Mathematical expression evaluation
- Safe calculation environment

**Tools:**
- `calculator`: Evaluates mathematical expressions

#### 3. **RagAgent** (`rag_agent.py`)
Handles document processing and knowledge base queries.

**Capabilities:**
- Document upload (PDF, TXT)
- Text chunking and splitting
- Vector embedding generation
- Semantic search
- Question answering based on documents

**Tools:**
- `add_document_to_knowledge`: Processes and stores documents
- `search_knowledge_base`: Searches for relevant information

### Memory System

#### **MemoryManager** (`memory_manager.py`)
Manages both short-term and long-term memory for the system.

**Short-term Memory:**
- Uses LangChain's `ConversationBufferMemory`
- Stores conversation history within a session
- Cleared when user clicks "Clear Chat"

**Long-term Memory:**
- Uses ChromaDB with Ollama embeddings
- Persistent storage across sessions
- Semantic search for relevant information

**Session Management:**
- Tracks session metadata
- Stores session IDs and timestamps
- Counts messages per session

#### **VectorStore** (`vector_store.py`)
Wrapper around ChromaDB for vector storage operations.

**Features:**
- Document embedding using Ollama
- Similarity search
- Persistent storage in local database
- Configurable collection names

### Tools

#### **Calculator** (`tools/calculator.py`)
Safe mathematical expression evaluator.

**Features:**
- Supports basic arithmetic operations (+, -, *, /)
- Handles parentheses and operator precedence
- Validates input for safety
- Error handling for invalid expressions

#### **Document Tools** (`tools/document.py`)
Utilities for document processing.

**Functions:**
- `document_loader`: Loads PDF and text files
- `split_document_content`: Splits documents into chunks with overlap

---

## ⚙️ Configuration

### Model Settings

The default configuration uses:
- **Model**: `llama3.2:3b`
- **Temperature**: 0.1 (more deterministic)
- **Max Tokens**: 1000

### Changing the Model

To use a different Ollama model, modify the model name in the relevant files:

**In `agents/ui_supervisor.py`:**
```python
def __init__(self, model_name="llama3.2:3b"):  # Change here
    self.llm = ChatOllama(model=model_name, temperature=0.1, max_tokens=1000)
```

**In `agents/base_agent.py` and `agents/rag_agent.py`:**
```python
self.llm = ChatOllama(model='llama3.2:3b', ...)  # Change here
```

**In `memory/memory_manager.py` and `memory/vector_store.py`:**
```python
self.embeddings = OllamaEmbeddings(model="llama3.2:3b")  # Change here
```

### Available Models

Popular Ollama models you can use:
- `llama3.2:3b` - Fast and efficient (default)
- `llama3.2:7b` - Better quality, slower
- `mistral` - Good alternative
- `codellama` - For coding tasks
- `phi` - Lightweight option

Pull a new model:
```bash
ollama pull <model-name>
```

### Adjusting Temperature

Temperature controls randomness (0.0 = deterministic, 2.0 = very creative):
```python
self.llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0.7,  # Adjust between 0.0 and 2.0
    max_tokens=1000
)
```

### Memory Configuration

**Persistent Directory:**
```python
MemoryManager(persist_directory="db")  # Change storage location
```

**Chunk Settings for Documents:**
```python
RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Size of each chunk
    chunk_overlap=200     # Overlap between chunks
)
```

---

## 🔮 Development Roadmap

### ✅ Completed
- Multi-agent architecture with specialized agents
- Dual memory system (short-term + long-term)
- Document upload and processing (PDF, TXT)
- Streamlit-based chat interface
- Smart query routing
- Calculator agent with safe evaluation
- RAG agent with vector search
- Session management

### 🚧 In Progress
- Additional supervisor implementations
- Enhanced error handling
- Performance optimizations

### 📋 Planned Features

**Phase 1: Core Enhancements**
- [ ] Enhanced error handling and logging
- [ ] Better session persistence
- [ ] Support for more document formats (DOCX, CSV, etc.)
- [ ] Configuration file support (YAML/JSON)
- [ ] Unit tests and integration tests

**Phase 2: Advanced Features**
- [ ] Web search agent integration
- [ ] Code execution agent
- [ ] Multi-modal support (images)
- [ ] Voice interface (speech-to-text)
- [ ] Advanced analytics and metrics

**Phase 3: Scalability & Integration**
- [ ] REST API for external integrations
- [ ] Plugin system for custom agents
- [ ] Multi-user support
- [ ] Database backend options (PostgreSQL, MongoDB)
- [ ] Export/import conversations

**Phase 4: Intelligence Improvements**
- [ ] Multi-language support
- [ ] Better context understanding
- [ ] Agent collaboration improvements
- [ ] Automated agent selection tuning
- [ ] Learning from user feedback

---

## 🔍 Troubleshooting

### Common Issues

#### **Issue: "Error connecting to Ollama"**

**Solution:**
1. Ensure Ollama is running: `ollama serve`
2. Check if the model is installed: `ollama list`
3. Pull the model if missing: `ollama pull llama3.2:3b`

#### **Issue: "Module not found" errors**

**Solution:**
```bash
pip install -r requirements.txt --upgrade
```

#### **Issue: Application is slow or crashes**

**Possible causes and solutions:**
1. **Insufficient RAM**: Close other applications or use a smaller model
2. **Large documents**: Split documents into smaller files before uploading
3. **Model size**: Switch to a smaller model like `llama3.2:3b` or `phi`

#### **Issue: Documents not being processed**

**Solution:**
1. Check file format is PDF or TXT
2. Ensure file is not corrupted
3. Check console for error messages
4. Verify file path in uploaded_docs directory

#### **Issue: Memory not persisting**

**Solution:**
1. Check if `db` directory exists and is writable
2. Verify session_metadata.json is being created
3. Ensure sufficient disk space

#### **Issue: Wrong agent is being called**

**Solution:**
This is a routing issue. Try rephrasing your query:
- For math: "Calculate X" or "What is X + Y"
- For documents: "What does my document say about X"
- For memory: "What did I ask before" or "Show conversation history"

### Debug Mode

To enable verbose logging, set `verbose=True` in agent executors:

```python
self.agent_executor = AgentExecutor(
    agent=self.agent, 
    tools=self.tools, 
    verbose=True  # Enable debug output
)
```

### Getting Help

If you encounter issues:
1. Check the console output for error messages
2. Review the [Issues](https://github.com/NIKHIL-DWIVEDI/AiAGENT/issues) page
3. Create a new issue with:
   - Error message
   - Steps to reproduce
   - System information (OS, Python version)

---

## 🤝 Contributing

Contributions are welcome! This project is under active development, and we'd love your help.

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

### Areas for Contribution

- Adding new agent types
- Improving existing agents
- Documentation improvements
- Bug fixes
- Test coverage
- Performance optimizations
- UI/UX enhancements

---

## 🙏 Acknowledgments

This project is built on top of excellent open-source tools:

- **[LangChain](https://python.langchain.com/)** - Framework for building LLM applications
- **[Ollama](https://ollama.ai/)** - Local LLM inference engine
- **[ChromaDB](https://www.trychroma.com/)** - Vector database for semantic search
- **[Streamlit](https://streamlit.io/)** - Web application framework
- **[LLaMA](https://ai.meta.com/llama/)** - Meta's open-source language models

### Special Thanks

- The LangChain community for excellent documentation
- Ollama team for making local LLM inference accessible
- All contributors to the open-source libraries used in this project

---

## 📄 License

This project is open source and available under the MIT License.

---

## 📧 Contact

**Project Maintainer**: Nikhil Dwivedi

**Repository**: [https://github.com/NIKHIL-DWIVEDI/AiAGENT](https://github.com/NIKHIL-DWIVEDI/AiAGENT)

---

## ⭐ Show Your Support

If you find this project useful, please consider:
- ⭐ Starring the repository
- 🐛 Reporting bugs
- 💡 Suggesting new features
- 🤝 Contributing to the code

**Made with ❤️ for the local AI community**