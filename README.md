# 🤖 Local AI Agent System

A sophisticated multi-agent AI system built with LangChain and Ollama that provides intelligent task routing, persistent memory, and document processing capabilities - all running locally for complete privacy.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)](https://python.langchain.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Features

- **🎯 Multi-Agent Architecture**: Specialized agents for different tasks (math, documents, memory)
- **🧠 Dual Memory System**: Short-term conversation memory + long-term semantic memory
- **📚 Document Intelligence**: Upload and query PDF/text documents with semantic search
- **🔒 Privacy-First**: Runs entirely locally - no data leaves your machine
- **💬 Interactive UI**: Beautiful Streamlit-based chat interface
- **🔄 Persistent Sessions**: Conversation history and user preferences are remembered
- **🧮 Smart Routing**: Automatically routes queries to the most appropriate agent

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE LAYER                     │
├─────────────────────────────────────────────────────────────┤
│  📱 Streamlit UI (app.py)                                   │
│  • Chat Interface                                           │
│  • File Upload                                              │
│  • Session Management                                       │
│  • System Metrics                                           │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                  ORCHESTRATION LAYER                        │
├─────────────────────────────────────────────────────────────┤
│  🎯 UISupervisor (ui_supervisor.py)                         │
│  • Main coordinator for UI requests                         │
│  • Routes queries to specialized agents                     │
│  • Manages conversation flow                                │
│  • Integrates memory system                                 │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                   SPECIALIZED AGENTS                        │
├─────────────────────────────────────────────────────────────┤
│  🧮 BaseAgent (Calculator)    📚 RagAgent (Documents)       │
│  • Math operations           • Document processing          │
│  • Calculations              • Knowledge retrieval          │
│  • Arithmetic                • PDF/Text handling            │
│                              • Vector search                │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    MEMORY SYSTEM                            │
├─────────────────────────────────────────────────────────────┤
│  🧠 MemoryManager (memory_manager.py)                       │
│  • Short-term: ConversationBufferMemory                     │
│  • Long-term: Vector-based persistent storage               │
│  • Session tracking & metadata                              │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                   STORAGE LAYER                             │
├─────────────────────────────────────────────────────────────┤
│  💾 VectorStore (ChromaDB)    📁 File System                │
│  • Document embeddings       • Session metadata             │
│  • Semantic search           • Uploaded files               │
│  • Knowledge persistence     • Configuration                │
└─────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Purpose | Key Features |
|-----------|---------|-------------|
| **UISupervisor** | Main orchestrator | Query routing, memory integration, response coordination |
| **BaseAgent** | Mathematics & calculations | Arithmetic operations, calculator tool integration |
| **RagAgent** | Document processing | PDF/text upload, semantic search, knowledge retrieval |
| **MemoryManager** | Conversation persistence | Short/long-term memory, session tracking |
| **VectorStore** | Knowledge storage | ChromaDB, embeddings, similarity search |

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) installed and running
- 4GB+ RAM recommended

### Installation

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install and start Ollama**
   ```bash
   # Install Ollama (if not already installed)
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Pull the required model
   ollama pull llama3.2:3b
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:8501`

## ⚙️ Configuration

### Model Settings
- **Default Model**: `llama3.2:3b`
- **Temperature**: 0.1 (deterministic responses)
- **Max Tokens**: 1000

### Customization
Edit the model settings in agent constructors:
```python
self.llm = ChatOllama(
    model="llama3.2:3b",  # Change model here
    temperature=0.1,       # Adjust creativity
    max_tokens=1000       # Response length limit
)
```

## 🔮 Roadmap

- [ ] **Web Search Integration**: Add real-time web search capabilities
- [ ] **Multi-Modal Support**: Image and audio processing agents
- [ ] **API Gateway**: REST API for external integrations
- [ ] **Plugin System**: Easy third-party agent integration
- [ ] **Advanced Analytics**: Usage patterns and performance metrics
- [ ] **Export/Import**: Conversation and knowledge base backup
- [ ] **Multi-Language**: Support for different languages
- [ ] **Voice Interface**: Speech-to-text and text-to-speech

## 🙏 Acknowledgments

- [LangChain](https://python.langchain.com/) for the agent framework
- [Ollama](https://ollama.ai/) for local LLM inference
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Streamlit](https://streamlit.io/) for the web interface

**Made with ❤️ for local AI enthusiasts**

⭐ Star this repository if you find it useful!