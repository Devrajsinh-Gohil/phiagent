import os
import streamlit as st
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
import sys
sys.path.append(str(Path(__file__).parent))

from rag_agent.rag_pipeline import RAGPipeline
from rag_agent.vector_store.pgvector_store import PgVectorStore
from rag_agent.config.settings import DB_CONFIG, RAG_CONFIG, UI_CONFIG
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Set page config
st.set_page_config(
    page_title=UI_CONFIG["page_title"],
    page_icon=UI_CONFIG["page_icon"],
    layout=UI_CONFIG["layout"],
    initial_sidebar_state=UI_CONFIG["initial_sidebar_state"]
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stApp {
        max-width: 1000px;
        margin: 0 auto;
    }
    .stTextInput>div>div>input {
        border-radius: 20px;
        padding: 10px 15px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
    }
    .chat-message.user {
        background-color: #f0f2f6;
        margin-left: 20%;
    }
    .chat-message.assistant {
        background-color: #e3f2fd;
        margin-right: 20%;
    }
    .message-content {
        margin-left: 1rem;
    }
    .source-badge {
        display: inline-block;
        background-color: #e0e0e0;
        padding: 0.2rem 0.5rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

class RAGApp:
    """Streamlit application for the RAG agent."""
    
    def __init__(self):
        """Initialize the RAG application."""
        self.initialize_session_state()
        self.setup_sidebar()
        self.setup_models()
    
    def initialize_session_state(self):
        """Initialize the Streamlit session state."""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        if 'rag_pipeline' not in st.session_state:
            st.session_state.rag_pipeline = None
        
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = []
    
    def setup_models(self):
        """Initialize the RAG pipeline with language models."""
        try:
            if st.session_state.rag_pipeline is None:
                with st.spinner("Initializing models..."):
                    # Initialize language models
                    llm = ChatOpenAI(
                        temperature=0.1,
                        model_name="gpt-4",
                        streaming=True
                    )
                    embeddings = OpenAIEmbeddings()
                    
                    # Initialize RAG pipeline
                    st.session_state.rag_pipeline = RAGPipeline(
                        llm=llm,
                        embeddings=embeddings,
                        vector_store=PgVectorStore()
                    )
        except Exception as e:
            st.error(f"Error initializing models: {str(e)}")
            logger.error(f"Error initializing models: {str(e)}")
    
    def setup_sidebar(self):
        """Set up the sidebar for file uploads and settings."""
        with st.sidebar:
            st.title("⚙️ Settings")
            
            # API Key input
            api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                value=os.getenv("OPENAI_API_KEY", ""),
                help="Get your API key from https://platform.openai.com/account/api-keys"
            )
            
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
            
            st.divider()
            
            # File upload
            st.subheader("📂 Upload Documents")
            uploaded_files = st.file_uploader(
                "Upload PDF, TXT, DOCX, or MD files",
                type=["pdf", "txt", "docx", "md"],
                accept_multiple_files=True
            )
            
            if uploaded_files:
                self.handle_file_uploads(uploaded_files)
            
            st.divider()
            
            # Session management
            st.subheader("💬 Session")
            
            # New chat button
            if st.button("New Chat"):
                st.session_state.messages = []
                if st.session_state.rag_pipeline:
                    st.session_state.rag_pipeline.clear_history()
                st.rerun()
            
            # Clear all data button
            if st.button("Clear All Data", type="secondary"):
                self.clear_all_data()
    
    def handle_file_uploads(self, uploaded_files):
        """Handle file uploads and process them."""
        if not uploaded_files:
            return
        
        # Get the list of newly uploaded files
        new_files = [
            file for file in uploaded_files 
            if file.name not in st.session_state.uploaded_files
        ]
        
        if not new_files:
            return
        
        # Save and process new files
        with st.spinner("Processing documents..."):
            for file in new_files:
                try:
                    # Save the uploaded file temporarily
                    file_path = Path("temp_uploads") / file.name
                    file_path.parent.mkdir(exist_ok=True)
                    
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                    
                    # Ingest the document
                    if st.session_state.rag_pipeline:
                        st.session_state.rag_pipeline.ingest_documents(str(file_path))
                        st.session_state.uploaded_files.append(file.name)
                        logger.info(f"Processed file: {file.name}")
                    
                    # Remove the temporary file
                    file_path.unlink()
                    
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
                    logger.error(f"Error processing {file.name}: {str(e)}")
            
            st.success(f"Processed {len(new_files)} document(s)")
    
    def clear_all_data(self):
        """Clear all data including chat history and vector store."""
        try:
            # Clear chat history
            st.session_state.messages = []
            
            # Clear vector store if it exists
            if st.session_state.rag_pipeline:
                st.session_state.rag_pipeline.vector_store.clear_collection()
            
            # Clear uploaded files list
            st.session_state.uploaded_files = []
            
            st.success("All data has been cleared.")
            st.rerun()
            
        except Exception as e:
            st.error(f"Error clearing data: {str(e)}")
            logger.error(f"Error clearing data: {str(e)}")
    
    def display_chat_messages(self):
        """Display chat messages from the session state."""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Display sources if available
                if message.get("sources"):
                    with st.expander("Sources"):
                        for source in message["sources"]:
                            st.markdown(f"- {source}")
    
    def handle_user_query(self, user_query: str):
        """Handle user query and display the response."""
        if not user_query.strip():
            return
        
        # Add user message to chat history
        st.session_state.messages.append({
            "role": "user",
            "content": user_query
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # Get response from RAG pipeline
                response = st.session_state.rag_pipeline.query(
                    user_query,
                    session_id="default"
                )
                
                # Display the response
                full_response = response["answer"]
                message_placeholder.markdown(full_response)
                
                # Display sources if available
                if response.get("sources"):
                    with st.expander("Sources"):
                        for source in response["sources"]:
                            st.markdown(f"- {source}")
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": response.get("sources", [])
                })
                
            except Exception as e:
                error_msg = "Sorry, I encountered an error processing your request."
                message_placeholder.error(error_msg)
                logger.error(f"Error generating response: {str(e)}")
                
                # Add error message to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "error": True
                })
    
    def run(self):
        """Run the Streamlit application."""
        st.title("🤖 RAG Chat Assistant")
        st.caption("Ask questions about your documents using the power of RAG")
        
        # Display chat messages
        self.display_chat_messages()
        
        # Chat input
        if prompt := st.chat_input("Ask me anything..."):
            self.handle_user_query(prompt)
            
            # Auto-scroll to bottom of chat
            st.rerun()

# Run the application
if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("temp_uploads", exist_ok=True)
    
    # Initialize and run the app
    app = RAGApp()
    app.run()
