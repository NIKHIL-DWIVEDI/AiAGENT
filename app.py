# app.py
import streamlit as st
import os
from agents.ui_supervisor import UISupervisor
import time

# Page configuration
st.set_page_config(
    page_title="Local AI Agent System",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'supervisor' not in st.session_state:
    with st.spinner("Initializing AI Agent System..."):
        st.session_state.supervisor = UISupervisor()

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Main title
st.title("🤖 Local AI Agent System")
st.markdown("*Powered by Local LLMs with Multi-Agent Architecture*")

# Sidebar
with st.sidebar:
    st.header("System Info")
    
    # Session info
    session_info = st.session_state.supervisor.get_session_info()
    st.metric("Messages in Session", session_info.get('messages', 0))
    
    # Capabilities
    st.header("🛠️ Available Agents")
    st.markdown("""
    - **🧮 Calculator**: Math and calculations
    - **📚 RAG**: Document Q&A and knowledge
    - **🧠 Memory**: Remember important information
    """)
    
    # File upload for RAG
    st.header("📄 Add Documents")
    uploaded_file = st.file_uploader(
        "Upload document for knowledge base",
        type=['txt', 'pdf'],
        help="Upload documents to add to the AI's knowledge base"
    )
    
    if uploaded_file:
        # Save uploaded file
        os.makedirs("uploaded_docs", exist_ok=True)
        file_path = os.path.join("uploaded_docs", uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Add to knowledge base
        with st.spinner(f"Adding {uploaded_file.name} to knowledge base..."):
            response = st.session_state.supervisor.run(f"Add the document {file_path} to your knowledge base")
            st.success(f"✅ Added {uploaded_file.name}")
    
    # Sample queries
    st.header("💡 Try These Examples")
    sample_queries = [
        "What is 25 * 17 + 8?",
        "Remember that I prefer Python programming",
        "What do you remember about me?",
        "Calculate 15% of 250"
    ]
    
    for query in sample_queries:
        if st.button(query, key=f"sample_{query[:10]}"):
            st.session_state.messages.append({"role": "user", "content": query})
            st.rerun()

# Main chat interface
st.header("💬 Chat with AI Agent")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("AI is thinking..."):
            response = st.session_state.supervisor.run(prompt)
        st.markdown(response)
    
    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})

# Clear chat button
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# Footer
st.markdown("---")
st.markdown("**Local AI Agent System** - Running entirely on your machine with privacy protection")