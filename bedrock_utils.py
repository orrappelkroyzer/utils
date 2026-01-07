"""
Amazon Bedrock utility functions for PDF embedding and Q&A retrieval.
Provides functions to embed PDFs from a directory and create a Q&A retriever.

Workflow for embedding once and querying in separate sessions:

Session 1 (Embed once):
    from pathlib import Path
    from utils.bedrock_utils import embed_pdfs_only
    
    embed_pdfs_only(
        pdf_directory=Path("path/to/pdfs"),
        persist_directory=Path("embeddings/my_docs")
    )

Session 2+ (Query multiple times):
    from pathlib import Path
    from utils.bedrock_utils import load_embeddings_and_create_qa
    
    qa = load_embeddings_and_create_qa(
        persist_directory=Path("embeddings/my_docs")
    )
    
    result = qa.query("What is the main topic?")
    print(result["answer"])
"""

import boto3
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import time
import sys
import pandas as pd
import numpy as np

# Add parent directory to path for config access
local_python_path = str(Path(__file__).parents[1])
if local_python_path not in sys.path:
    sys.path.append(local_python_path)
from utils.utils import load_config, get_logger

logger = get_logger(__name__)
config = load_config(add_date=False, config_path=Path(local_python_path) / 'config.json')

import fitz  # PyMuPDF

# NEW for LC 1.x
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

# Bedrock (preferred in 1.x)
# pip install langchain-aws
from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS



import faiss

# Text splitter (handle multiple package layouts)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Default models (GPT/OpenAI)
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_LLM_MODEL       = "gpt-5-thinking"

# Since we import required deps directly above, mark availability flags
LANGCHAIN_AVAILABLE = True
FAISS_AVAILABLE = True
PYMUPDF_AVAILABLE = True
try:
    from docx import Document  # python-docx
    DOCX_AVAILABLE = True
except Exception:
    DOCX_AVAILABLE = False


class BedrockPDFQA:
    """Class to handle PDF embedding and Q&A using Amazon Bedrock."""
    
    def __init__(self, 
                 region_name: str = "us-east-1",
                 embedding_model_id: str = DEFAULT_EMBEDDING_MODEL,
                 llm_model_id: str = DEFAULT_LLM_MODEL,
                 aws_access_key_id: Optional[str] = None,
                 aws_secret_access_key: Optional[str] = None):
        """
        Initialize Bedrock clients.
        
        Args:
            region_name: AWS region for Bedrock
            embedding_model_id: Bedrock embedding model ID
            llm_model_id: Bedrock LLM model ID
            aws_access_key_id: Optional AWS access key (uses credentials from environment if not provided)
            aws_secret_access_key: Optional AWS secret key
        """
        self.region_name = region_name
        self.embedding_model_id = embedding_model_id
        self.llm_model_id = llm_model_id
        
        # Initialize OpenAI (GPT) models
        if "OPENAI_API_KEY" not in os.environ and config.get("open_ai_key"):
            os.environ["OPENAI_API_KEY"] = str(config.get("open_ai_key"))
        self.embeddings = OpenAIEmbeddings(model=self.embedding_model_id)
        self.llm = ChatOpenAI(model=self.llm_model_id, temperature=0.1, max_tokens=4096)
                
        self.vector_store = None
        self.qa_chain = None
        self.documents = []
        
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF is required. Install with: pip install pymupdf")
        
        text = ""
        try:
            with fitz.open(pdf_path) as doc:
                for page_num, page in enumerate(doc):
                    page_text = page.get_text()
                    text += f"\n\n--- Page {page_num + 1} ---\n\n{page_text}"
            logger.info(f"Extracted {len(text)} characters from {pdf_path.name}")
            return text
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            raise

    def extract_text_from_docx(self, docx_path: Path) -> str:
        """
        Extract text from a DOCX file.
        """
        if not DOCX_AVAILABLE:
            raise ImportError("python-docx is required. Install with: pip install python-docx")
        try:
            doc = Document(docx_path)
            text = "\n".join(p.text for p in doc.paragraphs)
            logger.info(f"Extracted {len(text)} characters from {docx_path.name}")
            return text
        except Exception as e:
            logger.error(f"Error extracting text from {docx_path}: {e}")
            raise
    
    def process_pdfs_from_directory(self, 
                                    directory: Path,
                                    chunk_size: int = 1000,
                                    chunk_overlap: int = 200) -> List:
        """
        Process all PDFs in a directory and create document chunks.
        
        Args:
            directory: Directory containing PDF files
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of document chunks
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is required. Install with: pip install langchain langchain-community")
        
        directory = Path(directory)
        if not directory.exists():
            raise ValueError(f"Directory does not exist: {directory}")
        
        pdf_files = list(directory.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {directory}")
            return []
        
        logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
        all_documents = []
        
        for pdf_file in pdf_files:
            logger.info(f"Processing {pdf_file.name}...")
            try:
                text = self.extract_text_from_pdf(pdf_file)
                # Create document chunks
                chunks = text_splitter.create_documents([text])
                # Add metadata to each chunk
                for chunk in chunks:
                    chunk.metadata = {"source": str(pdf_file), "filename": pdf_file.name}
                all_documents.extend(chunks)
                logger.info(f"Created {len(chunks)} chunks from {pdf_file.name}")
            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}: {e}")
                continue
        
        self.documents = all_documents
        logger.info(f"Total chunks created: {len(all_documents)}")
        return all_documents

    def process_docx_from_directory(self,
                                    directory: Path,
                                    chunk_size: int = 1000,
                                    chunk_overlap: int = 200) -> List:
        """
        Process all DOCX files in a directory and create document chunks.
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is required. Install with: pip install langchain langchain-community")
        if not DOCX_AVAILABLE:
            raise ImportError("python-docx is required. Install with: pip install python-docx")

        directory = Path(directory)
        if not directory.exists():
            raise ValueError(f"Directory does not exist: {directory}")

        docx_files = list(directory.glob("*.docx"))
        if not docx_files:
            logger.warning(f"No DOCX files found in {directory}")
            return []

        logger.info(f"Found {len(docx_files)} DOCX files in {directory}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

        all_documents = []
        for docx_file in docx_files:
            logger.info(f"Processing {docx_file.name}...")
            try:
                text = self.extract_text_from_docx(docx_file)
                chunks = text_splitter.create_documents([text])
                for chunk in chunks:
                    chunk.metadata = {"source": str(docx_file), "filename": docx_file.name}
                all_documents.extend(chunks)
                logger.info(f"Created {len(chunks)} chunks from {docx_file.name}")
            except Exception as e:
                logger.error(f"Failed to process {docx_file.name}: {e}")
                continue

        self.documents = all_documents
        logger.info(f"Total chunks created: {len(all_documents)}")
        return all_documents
    
    def create_vector_store(self, documents: Optional[List] = None, persist_directory: Optional[Path] = None):
        """
        Create vector store from documents.
        
        Args:
            documents: List of document chunks (uses self.documents if not provided)
            persist_directory: Optional directory to save the vector store
        """
        if not LANGCHAIN_AVAILABLE or not FAISS_AVAILABLE:
            raise ImportError("LangChain and FAISS are required. Install with: pip install langchain langchain-community faiss-cpu")
        
        if documents is None:
            documents = self.documents
        
        if not documents:
            raise ValueError("No documents provided. Process PDFs first or provide documents.")
        
        logger.info("Creating vector store...")
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        
        if persist_directory:
            persist_dir = Path(persist_directory)
            persist_dir.mkdir(parents=True, exist_ok=True)
            self.vector_store.save_local(str(persist_dir))
            logger.info(f"Vector store saved to {persist_dir}")
    
    def load_vector_store(self, persist_directory: Path):
        """
        Load a saved vector store.
        
        Args:
            persist_directory: Directory where vector store is saved
        """
        if not LANGCHAIN_AVAILABLE or not FAISS_AVAILABLE:
            raise ImportError("LangChain and FAISS are required.")
        
        persist_dir = Path(persist_directory)
        if not persist_dir.exists():
            raise ValueError(f"Vector store directory does not exist: {persist_dir}")
        
        logger.info(f"Loading vector store from {persist_dir}")
        self.vector_store = FAISS.load_local(
            str(persist_dir),
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info("Vector store loaded successfully")
    
    def create_qa_chain(self, k: int = 3, search_type: str = "similarity"):
        if self.vector_store is None:
            raise ValueError("Vector store not created. Process PDFs and create vector store first.")

        retriever = self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )

        prompt = ChatPromptTemplate.from_template(
            "Use the following context to answer the question. "
            "If you don't know, say you don't know.\n\n"
            "Context:\n{context}\n\nQuestion: {input}\n\nAnswer:"
        )

        # LCEL pipeline: keep both answer and context
        # context gets Documents from retriever(question); answer is model output string
        self.qa_chain = (
            {"context": itemgetter("input") | retriever, "input": itemgetter("input")}
            | RunnablePassthrough.assign(
                answer=(prompt | self.llm | StrOutputParser())
            )
        )
    
    def query(self, question: str) -> Dict:
        """
        Query the Q&A system.
        
        Args:
            question: Question to ask
            
        Returns:
            Dictionary with 'answer' and 'source_documents'
        """
        if self.qa_chain is None:
            raise ValueError("Q&A chain not created. Run create_qa_chain() first.")
        
        logger.info(f"Query: {question}")
        result = self.qa_chain.invoke({"input": question})
        answer = result.get("answer", "")
        sources = result.get("context", [])
        return {
            "answer": answer,
            "source_documents": sources
        }
    
    def embed_pdfs_and_setup_qa(self,
                                pdf_directory: Path,
                                chunk_size: int = 1000,
                                chunk_overlap: int = 200,
                                k: int = 3,
                                persist_directory: Optional[Path] = None) -> "BedrockPDFQA":
        """
        Complete workflow: embed PDFs and set up Q&A.
        
        Args:
            pdf_directory: Directory containing PDF files
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            k: Number of documents to retrieve
            persist_directory: Optional directory to save/load vector store
            
        Returns:
            Self for method chaining
        """
        # Process PDFs
        self.process_pdfs_from_directory(
            pdf_directory,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Create vector store
        self.create_vector_store(persist_directory=persist_directory)
        
        # Create Q&A chain
        self.create_qa_chain(k=k)
        
        logger.info("PDF embedding and Q&A setup complete!")
        return self


def embed_pdfs_only(pdf_directory: Path,
                    persist_directory: Path,
                    region_name: str = "us-east-1",
                    embedding_model_id: str = DEFAULT_EMBEDDING_MODEL,
                    chunk_size: int = 1000,
                    chunk_overlap: int = 200,
                    aws_access_key_id: Optional[str] = None,
                    aws_secret_access_key: Optional[str] = None) -> Path:
    """
    Embed PDFs and save vector store (without creating Q&A). 
    Use this to embed once, then load in separate sessions for Q&A.
    
    Args:
        pdf_directory: Directory containing PDF files
        persist_directory: Directory to save the vector store (required)
        region_name: AWS region for Bedrock
        embedding_model_id: Bedrock embedding model ID
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        aws_access_key_id: Optional AWS access key
        aws_secret_access_key: Optional AWS secret key
        
    Returns:
        Path to the saved vector store
        
    Example:
        >>> # Session 1: Embed once
        >>> embed_pdfs_only(
        ...     pdf_directory=Path("path/to/pdfs"),
        ...     persist_directory=Path("embeddings/my_docs")
        ... )
        >>> 
        >>> # Session 2: Load and query (in separate script/session)
        >>> qa = load_embeddings_and_create_qa(
        ...     persist_directory=Path("embeddings/my_docs")
        ... )
        >>> result = qa.query("What is the main topic?")
    """
    if persist_directory is None:
        raise ValueError("persist_directory is required for embedding-only workflow")
    
    qa_system = BedrockPDFQA(
        region_name=region_name,
        embedding_model_id=embedding_model_id,
        llm_model_id=DEFAULT_LLM_MODEL,  # Not used, but required for initialization
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )
    
    # Process PDFs and create vector store (but don't create Q&A chain)
    qa_system.process_pdfs_from_directory(
        directory=pdf_directory,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    qa_system.create_vector_store(persist_directory=persist_directory)
    
    logger.info(f"PDFs embedded and saved to {persist_directory}. You can now load this in a separate session.")
    return Path(persist_directory)


def embed_docx_only(docx_directory: Path,
                    persist_directory: Path,
                    region_name: str = "eu-central-1",
                    embedding_model_id: str = DEFAULT_EMBEDDING_MODEL,
                    chunk_size: int = 1000,
                    chunk_overlap: int = 200,
                    aws_access_key_id: Optional[str] = None,
                    aws_secret_access_key: Optional[str] = None) -> Path:
    """
    Embed DOCX files and save vector store (without creating Q&A).
    """
    if persist_directory is None:
        raise ValueError("persist_directory is required for embedding-only workflow")

    qa_system = BedrockPDFQA(
        region_name=region_name,
        embedding_model_id=embedding_model_id,
        llm_model_id=DEFAULT_LLM_MODEL,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    qa_system.process_docx_from_directory(
        directory=docx_directory,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    qa_system.create_vector_store(persist_directory=persist_directory)
    logger.info(f"DOCX embedded and saved to {persist_directory}. You can now load this in a separate session.")
    return Path(persist_directory)


def load_embeddings_and_create_qa(persist_directory: Path,
                                   region_name: str = "us-east-1",
                                   embedding_model_id: str = DEFAULT_EMBEDDING_MODEL,
                                   llm_model_id: str = DEFAULT_LLM_MODEL,
                                   k: int = 3,
                                   aws_access_key_id: Optional[str] = None,
                                   aws_secret_access_key: Optional[str] = None) -> BedrockPDFQA:
    """
    Load pre-embedded vector store and create Q&A retriever.
    Use this in separate sessions after embedding PDFs with embed_pdfs_only().
    
    Args:
        persist_directory: Directory where vector store is saved (from embed_pdfs_only)
        region_name: AWS region for Bedrock
        embedding_model_id: Bedrock embedding model ID (must match the one used for embedding)
        llm_model_id: Bedrock LLM model ID
        k: Number of documents to retrieve
        aws_access_key_id: Optional AWS access key
        aws_secret_access_key: Optional AWS secret key
        
    Returns:
        BedrockPDFQA instance ready for queries
        
    Example:
        >>> # Load embeddings from previous session
        >>> qa = load_embeddings_and_create_qa(
        ...     persist_directory=Path("embeddings/my_docs")
        ... )
        >>> result = qa.query("What is the main topic?")
        >>> print(result["answer"])
    """
    qa_system = BedrockPDFQA(
        region_name=region_name,
        embedding_model_id=embedding_model_id,
        llm_model_id=llm_model_id,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )
    
    # Load existing vector store
    qa_system.load_vector_store(persist_directory=persist_directory)
    
    # Create Q&A chain
    qa_system.create_qa_chain(k=k)
    
    logger.info("Q&A system ready from loaded embeddings!")
    return qa_system


def embed_pdfs_and_create_qa(pdf_directory: Path,
                              region_name: str = "us-east-1",
                              embedding_model_id: str = DEFAULT_EMBEDDING_MODEL,
                              llm_model_id: str = DEFAULT_LLM_MODEL,
                              chunk_size: int = 1000,
                              chunk_overlap: int = 200,
                              k: int = 3,
                              persist_directory: Optional[Path] = None,
                              aws_access_key_id: Optional[str] = None,
                              aws_secret_access_key: Optional[str] = None) -> BedrockPDFQA:
    """
    Convenience function to embed PDFs and create Q&A retriever in one go.
    For embedding once and querying in separate sessions, use embed_pdfs_only() 
    and load_embeddings_and_create_qa() instead.
    
    Args:
        pdf_directory: Directory containing PDF files
        region_name: AWS region for Bedrock
        embedding_model_id: Bedrock embedding model ID
        llm_model_id: Bedrock LLM model ID
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        k: Number of documents to retrieve
        persist_directory: Optional directory to save vector store
        aws_access_key_id: Optional AWS access key
        aws_secret_access_key: Optional AWS secret key
        
    Returns:
        BedrockPDFQA instance ready for queries
        
    Example:
        >>> qa = embed_pdfs_and_create_qa(Path("path/to/pdfs"))
        >>> result = qa.query("What is the main topic?")
        >>> print(result["answer"])
    """
    qa_system = BedrockPDFQA(
        region_name=region_name,
        embedding_model_id=embedding_model_id,
        llm_model_id=llm_model_id,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )
    
    qa_system.embed_pdfs_and_setup_qa(
        pdf_directory=pdf_directory,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        k=k,
        persist_directory=persist_directory
    )
    
    return qa_system


def embed_texts_from_series(text_series: pd.Series,
                            embedding_model_id: str = DEFAULT_EMBEDDING_MODEL,
                            batch_size: int = 100) -> np.ndarray:
    """
    Embed texts from a pandas Series and return embeddings as numpy array.
    
    Args:
        text_series: pandas Series containing text strings to embed
        embedding_model_id: Embedding model ID to use
        region_name: AWS region (not used for OpenAI embeddings, kept for API consistency)
        aws_access_key_id: Optional AWS access key (not used for OpenAI)
        aws_secret_access_key: Optional AWS secret key (not used for OpenAI)
        batch_size: Number of texts to embed in each batch (for progress logging)
        
    Returns:
        numpy array of shape (n_texts, embedding_dim) containing embeddings
        
    Example:
        >>> import pandas as pd
        >>> texts = pd.Series(["Hello world", "How are you?", "Goodbye"])
        >>> embeddings = embed_texts_from_series(texts)
        >>> print(embeddings.shape)  # (3, embedding_dim)
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain is required. Install with: pip install langchain langchain-openai")
    
    # Initialize embeddings (using OpenAI)
    if "OPENAI_API_KEY" not in os.environ and config.get("open_ai_key"):
        os.environ["OPENAI_API_KEY"] = str(config.get("open_ai_key"))
    
    embeddings_model = OpenAIEmbeddings(model=embedding_model_id)
    
    # Filter out NaN/None values
    valid_mask = text_series.notna() & (text_series != '')
    valid_texts = text_series[valid_mask].tolist()
    
    if len(valid_texts) == 0:
        logger.warning("No valid texts found in series")
        return np.array([])
    
    logger.info(f"Embedding {len(valid_texts)} texts...")
    
    # Embed texts in batches for progress tracking
    all_embeddings = []
    for i in range(0, len(valid_texts), batch_size):
        logger.info(f"Embedding batch {i//batch_size} of {len(valid_texts)//batch_size} ({(i + batch_size)/len(valid_texts)*100:.2f}%)...")
        batch = valid_texts[i:i + batch_size]
        batch_embeddings = embeddings_model.embed_documents(batch)
        all_embeddings.extend(batch_embeddings)
        
        if (i + batch_size) % (batch_size * 10) == 0 or i + batch_size >= len(valid_texts):
            logger.info(f"Embedded {min(i + batch_size, len(valid_texts))}/{len(valid_texts)} texts...")
    
    embeddings_array = np.array(all_embeddings)
    logger.info(f"Embeddings completed. Shape: {embeddings_array.shape}")
    
    return embeddings_array

