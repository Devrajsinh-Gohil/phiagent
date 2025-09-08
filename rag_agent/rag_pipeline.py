import os
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from langchain_core.documents import Document as LangchainDocument
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from .vector_store.pgvector_store import PgVectorStore
from .config.settings import RAG_CONFIG
from .utils.document_processor import DocumentProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    """A RAG (Retrieval-Augmented Generation) pipeline for question answering."""
    
    def __init__(
        self,
        llm: BaseChatModel,
        embeddings: Embeddings,
        vector_store: Optional[PgVectorStore] = None,
        retriever: Optional[BaseRetriever] = None,
        prompt_template: Optional[str] = None,
        **kwargs
    ):
        """Initialize the RAG pipeline.
        
        Args:
            llm: Language model for generation
            embeddings: Embeddings model for document and query encoding
            vector_store: Vector store for document retrieval
            retriever: Pre-configured retriever (if None, will create one from vector_store)
            prompt_template: Custom prompt template for the RAG pipeline
        """
        self.llm = llm
        self.embeddings = embeddings
        self.vector_store = vector_store or PgVectorStore()
        
        # Set up retriever
        if retriever is None:
            self.retriever = self.vector_store.as_retriever(
                search_kwargs={"k": RAG_CONFIG["k"]}
            )
        else:
            self.retriever = retriever
        
        # Set up prompt template
        self.prompt_template = prompt_template or self._get_default_prompt()
        
        # Initialize the RAG chain
        self.chain = self._create_rag_chain()
        
        # Store chat histories
        self.chat_histories: Dict[str, ChatMessageHistory] = {}
    
    def _get_default_prompt(self) -> ChatPromptTemplate:
        """Get the default prompt template for the RAG pipeline."""
        return ChatPromptTemplate.from_messages([
            ("system", """

📊 Rajesh Javia Nifty 500 Valuation Model – Portfolio Advisor Prompt
You are a successful financial advisor in India who loves long-term equity mutual fund investing. You must behave like a professional advisor (clear, precise, friendly) and follow the Rajesh Javia Nifty 500 Valuation Model (TTM PE, PEG, CAPE estimate, ERP — Z-score removed).
Primary task: read the uploaded image (investor’s existing investments), read the uploaded Excel (universe + fund data), then run the workflow below. Use web lookups (factsheets, index data, G-sec) when needed and add citations. Always select recommended funds only from the uploaded Excel.
Inputs (required)
Image (uploaded): photo / pdf of investor holdings.
Excel file fund_universe.xlsx (columns: Fund Name, AMC, Category, Style, AUM, 1Y/3Y/5Y Returns, Beta, Expense Ratio, Top Holdings + weights, Cash & Equivalents %, Fund Manager, Mandate).
Workflow
Step 0 — Persona & tone
Friendly advisor tone: warm, slightly conversational, with small quotes.
Ask 3 profiling questions after parsing image:
Investment goal & horizon (Short <3y, Medium 3–7y, Long 7+y)
Risk tolerance (Low, Medium, High, Aggressive)
If market falls 20% from high — Sell / Hold SIPs / Invest more
Step 1 — Read image of holdings
OCR funds, units, values → produce neat table.
For each scheme: map Category & Style from Excel (or web factsheet if missing).
Validate style vs holdings → flag style drift.
Extract portfolio weights from Excel/factsheet.
Compute weight-based overlap matrix across all funds. Highlight >15%.
Step 2 — Market valuation (Rajesh Javia model)
Fetch: Nifty 500 TTM P/E, EPS estimates (FY25–26), expected EPS growth (~13% if no data), 10Y G-Sec yield, CAPE (proxy with Nifty 50 if needed).
Compute:
PE, PEG = PE / Growth,
Earnings Yield = 1/PE,
ERP = Earnings Yield – G-Sec yield,
CAPE estimate.
Classify zone:
Cheap → PE<18, PEG<1.2, ERP>1%
Fair → PE 18–21, PEG 1.2–1.5, ERP ~0–1%
Overvalued → PE>22, PEG>1.5, ERP<0%
Step 3 — Action rules by Valuation Zone
Cheap: Accumulate — overweight Momentum → Growth → Value.
Fair: Balanced — Growth → Value → Momentum → Quality.
Overvalued:
Tilt to Quality, Low-Beta, Contra/Value, Hybrid/Multi-asset.
Suggest hybrid/multi-asset for defensive allocation.
Fetch latest Cash & Equivalent % from AMC/factsheet (last month or month before).
Compare cash ratios across candidate funds. Prefer funds with higher defensive cash (>8–10%) when style/category same. Show in a comparison table.
Flag unusual high cash (>12–15%) and explain (tactical vs overly conservative).
Suggest redeem/switch 10–15% of equity from existing holdings (reduce beta/overlap). Provide staging plan (e.g., 5% now, 5% later).
Step 4 — Tax & friction caution
Warn against over-reshuffling (exit loads, LTCG/STCG).
For any suggested switch, estimate tax impact (approx).
Step 5 — Style diversification & due diligence
Ensure portfolio spans multiple styles: Growth, Value/Contra, Quality, Low-Beta, Momentum, Global (≤15%).
Due diligence checks (factsheet/web + citations):
Mandate vs actual holdings (flag drift)
Cash ratio >8–10% flagged (unless hybrid)
Top 5 holdings >40% flagged if mandate is “diversified”
Beta mismatch flagged
Step 6 — Keep scheme count low
SIP rules:
<₹2k → 1 fund
<₹5k → 1 fund
<₹10k → 1–2 funds
<₹25k → up to 4 funds
Any SIP >25k → max 5 funds
Lump sum → stagger (50% now, 50% later) if overvalued.
Step 7 — Scheme category priority
One scheme: Flexicap > Multicap > Hybrid > Multi-asset
Two: Flexicap/Multicap + Hybrid/Midcap/Contra/Smallcap
Three: Flexicap/Multicap + Midcap + Smallcap/Hybrid/Contra
Four: Flexicap/Multicap + Midcap/Index factor + Smallcap + Global (≤15%) + Hybrid + Contra
Four: Flexicap/Multicap + Midcap/factor + Smallcap + Global (≤15%) + Hybrid + Multi-asset + one Thematic/Contra
Step 8 — Output format
A. Executive Summary (3 lines)
B. Investor profile (from Qs)
C. Parsed holdings table
D. Fund style mapping & drift flags
E. Overlap matrix
F. Market metrics (PE, PEG, ERP, CAPE, zone + citations)
G. Action Plan (rebalance, SIP/Lumpsum suggestions, staging, tax note)
H. Risk notes (monitor, rebalancing triggers)
I. Investor checklist (6 items)
J. JSON payload (profile, holdings[], overlap_matrix, metrics, actions[], sip_plan[], citations[])
Rules
Only recommend from Excel (unless investor allows adding new).
Overlap must be weight-based (Σ min(wA, wB)).
Cash ratio used in Overvalued zone to compare candidate funds.
Always cite sources (AMC/factsheet/web).
Max global exposure 15%.
Friendly but professional tone; include small motivational quotes.
            
            Context: {context}
            
            Question: {input}
            
            Answer:"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
    
    def _create_rag_chain(self):
        """Create the RAG chain with history awareness."""
        # Create a chain that answers with context
        document_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=self.prompt_template,
            output_parser=StrOutputParser()
        )
        
        # Create a retriever chain that can use conversation history
        retriever_chain = create_history_aware_retriever(
            llm=self.llm,
            retriever=self.retriever,
            prompt=ChatPromptTemplate.from_messages([
                MessagesPlaceholder("chat_history"),
                ("user", "{input}"),
                ("user", """Given the above conversation, generate a search query to look up in order to get information relevant to the conversation""")
            ])
        )
        
        # Combine the retriever and document chain
        rag_chain = create_retrieval_chain(retriever_chain, document_chain)
        
        # Add message history
        chain_with_history = RunnableWithMessageHistory(
            rag_chain,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
        
        return chain_with_history
    
    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Get or create chat history for a session."""
        if session_id not in self.chat_histories:
            self.chat_histories[session_id] = ChatMessageHistory()
        return self.chat_histories[session_id]
    
    def ingest_documents(
        self, 
        file_paths: Union[str, List[str]], 
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        **kwargs
    ) -> List[str]:
        """Ingest documents into the vector store.
        
        Args:
            file_paths: Single file path or list of file paths to ingest
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of document IDs that were added
        """
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        
        # Process each file
        all_docs = []
        processor = DocumentProcessor()
        
        for file_path in file_paths:
            file_path = Path(file_path)
            if file_path.is_file():
                # Process single file
                content = processor.load_document(str(file_path))
                doc = LangchainDocument(
                    page_content=content,
                    metadata={
                        "source": str(file_path.name),
                        "file_path": str(file_path),
                    }
                )
                all_docs.append(doc)
            elif file_path.is_dir():
                # Process directory
                docs = processor.process_directory(
                    str(file_path),
                    chunk_size=chunk_size or RAG_CONFIG["chunk_size"],
                    chunk_overlap=chunk_overlap or RAG_CONFIG["chunk_overlap"]
                )
                all_docs.extend(docs)
        
        # Add documents to vector store
        if all_docs:
            return self.vector_store.add_documents(
                documents=all_docs,
                embeddings=self.embeddings,
                **kwargs
            )
        return []
    
    def query(
        self, 
        question: str, 
        session_id: str = "default",
        **kwargs
    ) -> Dict[str, Any]:
        """Query the RAG pipeline.
        
        Args:
            question: The question to ask
            session_id: Session ID for maintaining conversation history
            
        Returns:
            Dictionary containing the answer and source documents
        """
        try:
            response = self.chain.invoke(
                {"input": question},
                config={"configurable": {"session_id": session_id}}
            )
            
            # Format the response
            result = {
                "answer": response["answer"],
                "sources": list(set(
                    doc.metadata.get("source", "Unknown") 
                    for doc in response.get("context", [])
                )),
                "context": response.get("context", [])
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error querying RAG pipeline: {str(e)}")
            return {
                "answer": "I'm sorry, I encountered an error processing your request.",
                "sources": [],
                "context": []
            }
    
    def clear_history(self, session_id: Optional[str] = None) -> None:
        """Clear chat history for a session or all sessions.
        
        Args:
            session_id: Session ID to clear (if None, clears all sessions)
        """
        if session_id is None:
            self.chat_histories.clear()
            logger.info("Cleared all chat histories")
        elif session_id in self.chat_histories:
            self.chat_histories[session_id].clear()
            logger.info(f"Cleared chat history for session: {session_id}")

# Example usage
if __name__ == "__main__":
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    
    # Initialize models
    llm = ChatOpenAI(temperature=0.1, model_name="gpt-4")
    embeddings = OpenAIEmbeddings()
    
    # Initialize RAG pipeline
    rag = RAGPipeline(llm=llm, embeddings=embeddings)
    
    # Example: Ingest documents
    # doc_ids = rag.ingest_documents("path/to/your/documents")
    # print(f"Ingested {len(doc_ids)} document chunks")
    
    # Example: Query the RAG pipeline
    response = rag.query("What is artificial intelligence?")
    print(f"Answer: {response['answer']}")
    print(f"Sources: {', '.join(response['sources'])}")
