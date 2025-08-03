# app.py
import streamlit as st
import os
from agents.ui_supervisor import UISupervisor
import time

# Page configuration
st.set_page_config(
    page_title="Personal AI Agent System",
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
st.title("🤖 Personal AI Agent System")

# Sidebar
with st.sidebar:
    # st.header("System Info")
    
    # Session info
    # session_info = st.session_state.supervisor.get_session_info()
    # st.metric("Messages in Session", session_info.get('messages', 0))
    
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
        
        # Add to knowledge base with proper command
        with st.spinner(f"Adding {uploaded_file.name} to knowledge base..."):
            # Use the exact file path that was saved
            response = st.session_state.supervisor.run(f"Add the document at path {file_path} to your knowledge base")
            
            if "Successfully processed" in response or "success" in response.lower():
                st.success(f"✅ Added {uploaded_file.name}")
                st.info(f"File saved at: {file_path}")
            else:
                st.error(f"❌ Failed to add {uploaded_file.name}")
                st.error(f"Error: {response}")
                
        # Clear the uploader to prevent re-processing
        st.rerun()
    


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
    st.session_state
    st.session_state.supervisor.memory_manager.clear_conversation_history()
    st.success("Chat cleared successfully!")
    st.rerun()
