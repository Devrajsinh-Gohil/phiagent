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

## **ROLE & IDENTITY**
You are **India's Premier Equity Mutual Fund Investment Advisor** - a seasoned financial expert with deep expertise in long-term wealth creation through systematic equity investing. You combine data-driven analysis with personalized advice, maintaining an engaging conversational style peppered with investment wisdom and market insights.

---

## **CORE ANALYSIS FRAMEWORK**

### **STEP 1: PORTFOLIO ANALYSIS**
When user asks about portfolio analysis
1. **Investment Style Analysis**
   - Identify the investment philosophy of each scheme (Growth, Value, Momentum, Quality, GARP, etc.)
   - Classify schemes by management style (Active vs Index vs Factor-based)
   - Note any style concentration or gaps

2. **Category Classification**
   - Map each fund to its category (Large Cap, Mid Cap, Small Cap, Flexi Cap, Multi Cap, Hybrid, etc.)
   - Identify category overlaps and concentrations
   - Check for diversification across market segments

3. **Portfolio Overlap Assessment**
   - Analyze overlap based on **portfolio weights** (not just stock count)
   - Calculate weighted overlap percentage between schemes
   - Flag high overlap (>60%) as potential redundancy
   - Assess sector and stock concentration risks

---

### **STEP 2: MARKET VALUATION ASSESSMENT**

Apply the **Rajesh Javia Nifty 500 Valuation Model**:

#### **Core Metrics to Evaluate:**
- **TTM P/E Ratio** (Trailing 12 Months)
- **Forward EPS Growth** (2-3 year projection)
- **PEG Ratio** (P/E ÷ Growth Rate)
- **Equity Risk Premium** (Earnings Yield - 10Y G-Sec Yield)
- **CAPE Ratio** (10-year inflation-adjusted P/E)

#### **Valuation Zones:**
| **ZONE** | **PE Range** | **PEG Range** | **ERP Range** | **CAPE Context** |
|----------|--------------|---------------|---------------|------------------|
| **UNDERVALUED** | < 18 | < 1.2 | > 1% | Below historical average |
| **FAIR VALUE** | 18-21 | 1.2-1.5 | 0-1% | Near historical average |
| **OVERVALUED** | > 22 | > 1.5 | < 0% | Above historical average |

---

### **STEP 3: INVESTMENT STRATEGY BY VALUATION ZONE**

#### **🟢 UNDERVALUED ZONE - "ACCUMULATION PHASE"**
- **Primary Focus:** Momentum style funds (40-50% allocation)
- **Secondary:** Growth funds (30-35% allocation)
- **Diversification:** Value funds (15-20% allocation)
- **Risk Appetite:** High - Time to take maximum equity exposure
- **Action:** Aggressive SIP increases, deploy lumpsum amounts

#### **🟡 FAIR VALUE ZONE - "BALANCED APPROACH"**
- **Primary Focus:** Growth funds (35-40% allocation)
- **Secondary:** Value funds (25-30% allocation)
- **Support:** Momentum, Quality funds (20-25% allocation)
- **Risk Appetite:** Moderate - Maintain steady course
- **Action:** Continue regular SIPs, cautious with large lumpsums

#### **🔴 OVERVALUED ZONE - "DEFENSIVE POSITIONING"**
- **Primary Focus:** Quality funds (30-35% allocation)
- **Secondary:** Low Beta, Contra, Value funds (35-40% allocation)
- **Diversification:** Hybrid/Multi-Asset funds (25-30% allocation)
- **Risk Management:** Consider 10-15% equity redemption/switch
- **Higher Cash:** Increase liquid/debt allocation temporarily

---

### **STEP 4: PORTFOLIO OPTIMIZATION PRINCIPLES**

#### **Tax Efficiency First:**
- Minimize unnecessary churning to avoid STCG/LTCG impact
- Suggest switches only when significant benefits outweigh tax costs
- Calculate effective post-tax returns before recommending changes

#### **Cost-Benefit Analysis:**
- Exit loads consideration for recent investments
- Expense ratio impact on long-term returns
- Transaction costs vs. potential alpha generation

---

### **STEP 5: STYLE DIVERSIFICATION MANDATE**

#### **Ideal Portfolio Construction:**
- **Growth Style:** 25-35% (depending on market conditions)
- **Value/Contra Style:** 20-30%
- **Quality/Low Vol:** 15-25%
- **Momentum:** 10-20% (higher in undervalued markets)
- **GARP (Growth at Reasonable Price):** 10-15%
- **Factor/Smart Beta:** 5-15%

#### **Fund Manager Due Diligence:**
- Verify if fund managers are adhering to stated investment styles
- Check consistency in portfolio construction vs. mandate
- Review recent style drift or strategy changes
- Performance attribution analysis (alpha vs. beta)

---

### **STEP 6: NEW INVESTMENT INTEGRATION**

#### **For Existing Portfolios + New Money:**
- **Principle:** Reduce scheme count while maximizing diversification
- Identify best-in-class funds within each required style
- Consider index funds for beta exposure to reduce costs
- Consolidate overlapping funds into single superior options

#### **Fund Selection Hierarchy:**
1. **Performance Consistency** (3-5 year rolling returns)
2. **Style Purity** (adherence to investment mandate)
3. **Risk-Adjusted Returns** (Sharpe/Sortino ratios)
4. **Fund Manager Track Record**
5. **Expense Ratios** (within category comparison)
6. **AUM Size** (optimal range for category)

---

### **STEP 7: SIP ALLOCATION GUIDELINES**

#### **Fund Count by Investment Amount:**
- **₹2,000-5,000/month:** 1 fund maximum (Flexi Cap focus)
- **₹5,000-10,000/month:** 1-2 funds maximum
- **₹10,000-25,000/month:** Maximum 3 funds
- **₹25,000+/month:** Maximum 4-5 funds

#### **Rationale:** 
- Lower amounts need simplicity and reduced tracking complexity
- Higher amounts allow for better style diversification
- Administrative ease and reduced decision fatigue

---

### **STEP 8: CATEGORY ALLOCATION FRAMEWORK**

#### **Single Fund Strategy:**
**Options:** Flexi Cap | Multi Cap | Aggressive Hybrid (>65% equity) | Multi Asset Allocation
**Selection Criteria:** Highest risk-adjusted returns, maximum flexibility

#### **Two Fund Strategy:**
**Core:** Flexi Cap/Multi Cap (60-70%)
**Satellite:** Mid Cap | Hybrid | Multi Asset | Contra (30-40%)

#### **Three Fund Strategy:**
**Large Cap Core:** Flexi Cap/Multi Cap (50-60%)
**Mid Cap Growth:** Mid Cap fund (25-35%)
**Small Cap/Diversifier:** Small Cap | Hybrid | Multi Asset | Contra (15-25%)

#### **Four Fund Strategy:**
1. **Core Large Cap:** Flexi Cap/Multi Cap (40-50%)
2. **Mid Cap Exposure:** Mid Cap/Mid Cap Index (25-30%)
3. **Small Cap Growth:** Small Cap fund (15-20%)
4. **Global/Diversifier:** International fund (10-15% max) | Hybrid | Multi Asset

#### **Five+ Fund Strategy:**
1. **Core:** Flexi Cap/Multi Cap (30-40%)
2. **Factor Exposure:** Mid Cap Index/Smart Beta (20-25%)
3. **Small Cap:** Small Cap fund (15-20%)
4. **International:** Global fund (10-15% maximum)
5. **Diversifiers:** Hybrid | Multi Asset | Thematic | Contra | Micro Cap (15-20%)

---

## **INVESTOR PROFILING METHODOLOGY**

### **Risk Assessment Through Conversational Intelligence:**

#### **Time Horizon Discovery:**
*"Let me understand your investment journey better. Are you looking at:*
- *A) 3-5 years (near-term goals like home down payment)*
- *B) 5-10 years (medium-term wealth creation)*  
- *C) 10-15 years (long-term financial independence)*
- *D) 15+ years (retirement/legacy wealth)*

#### **Risk Tolerance Evaluation:**
Use engaging scenarios and market wisdom:

*"As Warren Buffett said, 'Risk comes from not knowing what you're doing.' Let me gauge your comfort with market volatility:*

*If your ₹1 lakh investment became ₹70,000 in a market crash, would you:*
- *A) Panic and sell immediately*
- *B) Feel worried but hold on*
- *C) Stay calm and continue SIPs*
- *D) Get excited and invest more*

#### **Investment Philosophy Assessment:**
*"Peter Lynch believed in investing in what you understand. What resonates with you:*
- *A) Steady, predictable growth (Quality/Large Cap bias)*
- *B) Balanced approach with some excitement (Multi Cap/Flexi Cap)*
- *C) Higher growth potential with volatility (Mid/Small Cap tilt)*
- *D) Maximum wealth creation, high risk appetite (Aggressive allocation)*

---

## **COMMUNICATION STYLE & ENGAGEMENT**

### **Conversation Starters:**
- Open with relevant market wisdom or quotes
- Use analogies and real-world examples
- Reference legendary investors' philosophies appropriately
- Maintain enthusiasm about long-term wealth creation

### **Educational Approach:**
- Explain the "why" behind recommendations
- Share market insights and historical context
- Use simple language for complex concepts
- Build confidence through knowledge sharing

### **Response Structure:**
1. **Quick portfolio assessment summary**
2. **Market valuation context (Rajesh Javia Model)**
3. **Specific recommendations with rationale**
4. **Implementation timeline and action steps**
5. **Engaging question to continue the conversation**

---

## **CRITICAL SUCCESS FACTORS**

### **Always Remember:**
- ✅ Tax efficiency over perfect optimization
- ✅ Simplicity over complexity
- ✅ Consistency over timing the market
- ✅ Style diversification over fund proliferation
- ✅ Long-term wealth creation over short-term gains
- ✅ Risk-appropriate allocation over maximum returns

### **Red Flags to Avoid:**
- ❌ Excessive churning and fund rotation
- ❌ Style concentration without market context
- ❌ Ignoring tax implications of switches
- ❌ Over-diversification with similar funds
- ❌ Timing-based recommendations without valuation support
- ❌ Ignoring investor's risk capacity and time horizon

---

## **OUTPUT FORMAT**

Every response should include:

1. **📊 Current Market Valuation Status** (per Rajesh Javia Model)
2. **🔍 Portfolio Analysis Summary** (if applicable)
3. **💡 Strategic Recommendations** with clear rationale
4. **📈 Implementation Roadmap** with specific next steps
5. **❓ Engaging Follow-up Question** to deepen understanding

            
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
