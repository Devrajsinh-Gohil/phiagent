import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Database Configuration
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "rag_database"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
}

# Vector Store Configuration
VECTOR_STORE_CONFIG = {
    "collection_name": "document_embeddings",
    "embedding_dimension": 1536,  # OpenAI text-embedding-ada-002 dimension
    "distance_strategy": "cosine",
}

# LLM Configuration
LLM_CONFIG = {
    "model_name": "gpt-4",
    "temperature": 0.1,
    "max_tokens": 1000,
}

# Embeddings Configuration
EMBEDDING_CONFIG = {
    "model_name": "text-embedding-ada-002",
    "model_kwargs": {"device": "cpu"},
}

# RAG Configuration
RAG_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "k": 4,  # Number of documents to retrieve
}

# Streamlit UI Configuration
UI_CONFIG = {
    "page_title": "RAG Agent with LangGraph",
    "page_icon": "🤖",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
VECTOR_STORE_DIR = os.path.join(ROOT_DIR, "vector_store")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
