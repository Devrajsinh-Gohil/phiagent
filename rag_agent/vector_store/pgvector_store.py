import os
import logging
from typing import List, Dict, Any, Optional, Union
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
import numpy as np
from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document as LangchainDocument
from langchain_community.embeddings import OpenAIEmbeddings

from ..config.settings import DB_CONFIG, VECTOR_STORE_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PgVectorStore:
    """A wrapper around pgvector for vector similarity search."""
    
    def __init__(self, connection_string: Optional[str] = None, collection_name: Optional[str] = None):
        """Initialize the PgVectorStore.
        
        Args:
            connection_string: PostgreSQL connection string
            collection_name: Name of the collection/table to store vectors
        """
        self.connection_string = connection_string or self._get_connection_string()
        self.collection_name = collection_name or VECTOR_STORE_CONFIG["collection_name"]
        self.embedding_dimension = VECTOR_STORE_CONFIG["embedding_dimension"]
        self.distance_strategy = VECTOR_STORE_CONFIG["distance_strategy"]
        self._initialize_database()
    
    def _get_connection_string(self) -> str:
        """Construct the connection string from environment variables."""
        return (
            f"postgresql+psycopg2://{DB_CONFIG['user']}:"
            f"{DB_CONFIG['password']}@{DB_CONFIG['host']}:"
            f"{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
        )
    
    def _initialize_database(self) -> None:
        """Initialize the database with pgvector extension if not exists."""
        try:
            conn = psycopg2.connect(
                dbname=DB_CONFIG["dbname"],
                user=DB_CONFIG["user"],
                password=DB_CONFIG["password"],
                host=DB_CONFIG["host"],
                port=DB_CONFIG["port"]
            )
            conn.autocommit = True
            cursor = conn.cursor()
            
            # Enable pgvector extension
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # Create table if not exists
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {self.collection_name} (
                id SERIAL PRIMARY KEY,
                content TEXT,
                metadata JSONB,
                embedding VECTOR({self.embedding_dimension}),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
            """
            cursor.execute(create_table_query)
            
            # Create index for faster similarity search
            index_name = f"idx_{self.collection_name}_embedding"
            cursor.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {index_name} 
                ON {self.collection_name} 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
                """
            )
            
            cursor.close()
            conn.close()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise
    
    def as_retriever(self, **kwargs):
        """Get a retriever for this vector store."""
        return self.vector_store.as_retriever(**kwargs)
    
    def add_documents(
        self, 
        documents: List[LangchainDocument],
        embeddings: Optional[Embeddings] = None,
        **kwargs
    ) -> List[str]:
        """Add documents to the vector store.
        
        Args:
            documents: List of Langchain Document objects
            embeddings: Embeddings model to use for generating vector representations
            
        Returns:
            List of document IDs that were added
        """
        if not documents:
            return []
            
        embeddings = embeddings or OpenAIEmbeddings()
        
        # Create LangChain PGVector instance
        vector_store = PGVector.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=self.collection_name,
            connection_string=self.connection_string,
            pre_delete_collection=False,  # Don't delete existing data
            **kwargs
        )
        
        # Get the document IDs that were added
        # This is a workaround since PGVector doesn't return the IDs directly
        doc_ids = [str(hash(doc.page_content + str(doc.metadata))) for doc in documents]
        
        logger.info(f"Added {len(documents)} documents to the vector store")
        return doc_ids
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 4, 
        filter: Optional[Dict[str, Any]] = None,
        embeddings: Optional[Embeddings] = None,
        **kwargs
    ) -> List[LangchainDocument]:
        """Perform similarity search.
        
        Args:
            query: Query text
            k: Number of results to return
            filter: Optional filter to apply to metadata
            embeddings: Embeddings model to use for query embedding
            
        Returns:
            List of documents most similar to the query
        """
        embeddings = embeddings or OpenAIEmbeddings()
        
        # Create LangChain PGVector instance
        vector_store = PGVector(
            collection_name=self.collection_name,
            connection_string=self.connection_string,
            embedding_function=embeddings,
            **kwargs
        )
        
        return vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter
        )
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 4, 
        filter: Optional[Dict[str, Any]] = None,
        embeddings: Optional[Embeddings] = None,
        **kwargs
    ) -> List[tuple[LangchainDocument, float]]:
        """Perform similarity search with scores.
        
        Args:
            query: Query text
            k: Number of results to return
            filter: Optional filter to apply to metadata
            embeddings: Embeddings model to use for query embedding
            
        Returns:
            List of (document, score) tuples
        """
        embeddings = embeddings or OpenAIEmbeddings()
        
        # Create LangChain PGVector instance
        vector_store = PGVector(
            collection_name=self.collection_name,
            connection_string=self.connection_string,
            embedding_function=embeddings,
            **kwargs
        )
        
        return vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter
        )
    
    def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents by their IDs.
        
        Args:
            document_ids: List of document IDs to delete
        """
        if not document_ids:
            return
            
        try:
            conn = psycopg2.connect(
                dbname=DB_CONFIG["dbname"],
                user=DB_CONFIG["user"],
                password=DB_CONFIG["password"],
                host=DB_CONFIG["host"],
                port=DB_CONFIG["port"]
            )
            conn.autocommit = True
            cursor = conn.cursor()
            
            # Delete documents with matching IDs
            placeholders = ','.join(['%s'] * len(document_ids))
            cursor.execute(
                f"""
                DELETE FROM {self.collection_name}
                WHERE id = ANY(%s)
                """,
                (document_ids,)
            )
            
            logger.info(f"Deleted {cursor.rowcount} documents from the vector store")
            cursor.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            raise
    
    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        try:
            conn = psycopg2.connect(
                dbname=DB_CONFIG["dbname"],
                user=DB_CONFIG["user"],
                password=DB_CONFIG["password"],
                host=DB_CONFIG["host"],
                port=DB_CONFIG["port"]
            )
            conn.autocommit = True
            cursor = conn.cursor()
            
            cursor.execute(f"TRUNCATE TABLE {self.collection_name}")
            logger.info(f"Cleared all documents from collection: {self.collection_name}")
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            raise
            
    def as_retriever(self, **kwargs):
        """Return a retriever for this vector store.
        
        Args:
            **kwargs: Additional arguments to pass to the retriever
                - search_kwargs: Dictionary of search parameters (e.g., {"k": 5})
                
        Returns:
            A retriever for this vector store
        """
        from langchain_community.vectorstores.pgvector import PGVector
        
        # Extract search_kwargs if present
        search_kwargs = kwargs.pop('search_kwargs', {})
        
        # Create a new PGVector instance for retrieval
        vector_store = PGVector(
            collection_name=self.collection_name,
            connection_string=self.connection_string,
            embedding_function=OpenAIEmbeddings()
        )
        
        # Pass search_kwargs to the retriever
        return vector_store.as_retriever(search_kwargs=search_kwargs, **kwargs)

# Example usage
if __name__ == "__main__":
    # Initialize the vector store
    vector_store = PgVectorStore()
    
    # Example document
    documents = [
        LangchainDocument(
            page_content="This is a test document about artificial intelligence.",
            metadata={"source": "test", "page": 1}
        )
    ]
    
    # Add documents
    doc_ids = vector_store.add_documents(documents)
    print(f"Added documents with IDs: {doc_ids}")
    
    # Search for similar documents
    query = "What is AI?"
    results = vector_store.similarity_search(query, k=2)
    print(f"Search results for '{query}':")
    for i, doc in enumerate(results):
        print(f"{i+1}. {doc.page_content[:100]}...")
    
    # Clean up
    vector_store.clear_collection()
