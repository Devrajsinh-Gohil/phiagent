# RAG Agent with LangChain, LangGraph, and Streamlit

A powerful Retrieval-Augmented Generation (RAG) agent built with LangChain, LangGraph, and Streamlit, using pgvector for vector storage.

## Features

- **Document Processing**: Supports PDF, TXT, DOCX, and Markdown files
- **Vector Storage**: Utilizes pgvector for efficient similarity search
- **Conversational AI**: Maintains conversation history for context-aware responses
- **User-Friendly Interface**: Built with Streamlit for an intuitive web interface
- **Modular Design**: Easy to extend and customize

## Prerequisites

- Python 3.8+
- PostgreSQL with pgvector extension
- OpenAI API key

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd phiagent
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the project root with the following variables:
   ```
   OPENAI_API_KEY=your_openai_api_key
   DB_NAME=rag_database
   DB_USER=postgres
   DB_PASSWORD=postgres
   DB_HOST=localhost
   DB_PORT=5432
   ```

5. **Set up PostgreSQL with pgvector**
   - Install PostgreSQL if you haven't already
   - Install the pgvector extension:
     ```sql
     CREATE EXTENSION vector;
     ```
   - Create a database for the RAG agent:
     ```sql
     CREATE DATABASE rag_database;
     ```

## Usage

1. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

2. **Using the application**
   - Upload documents using the sidebar
   - Ask questions in the chat interface
   - Start a new chat or clear all data using the sidebar controls

## Project Structure

```
phiagent/
├── app.py                  # Streamlit application
├── requirements.txt        # Python dependencies
├── .env.example           # Example environment variables
├── rag_agent/
│   ├── __init__.py
│   ├── config/
│   │   └── settings.py    # Configuration settings
│   ├── utils/
│   │   └── document_processor.py  # Document processing utilities
│   ├── vector_store/
│   │   └── pgvector_store.py     # pgvector integration
│   └── rag_pipeline.py    # RAG pipeline implementation
└── data/                  # Directory for document storage
```

## Customization

### Adding New Document Types

To add support for additional document types:

1. Update the `DocumentProcessor` class in `rag_agent/utils/document_processor.py`
2. Add a new method to handle the file type
3. Update the `load_document` method to use the new handler

### Modifying the RAG Pipeline

The RAG pipeline can be customized by modifying the `RAGPipeline` class in `rag_agent/rag_pipeline.py`. You can:

- Change the prompt template
- Adjust chunking parameters
- Add custom retrieval or generation logic

## Troubleshooting

### Common Issues

1. **pgvector extension not found**
   - Ensure the pgvector extension is installed in your PostgreSQL database
   - Run `CREATE EXTENSION IF NOT EXISTS vector;` in your database

2. **API key not found**
   - Make sure you've set the `OPENAI_API_KEY` in your `.env` file

3. **Database connection issues**
   - Verify your database credentials in the `.env` file
   - Ensure PostgreSQL is running and accessible

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LangChain](https://langchain.com/)
- [LangGraph](https://langchain.com/langgraph/)
- [Streamlit](https://streamlit.io/)
- [pgvector](https://github.com/pgvector/pgvector)
