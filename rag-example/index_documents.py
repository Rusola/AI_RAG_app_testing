"""
Index source documents and persist in vector embedding database.

This script processes PDF documents and creates a local vector database (Chroma)
that stores semantic embeddings of the document content. These embeddings allow
for intelligent semantic search and retrieval-augmented generation (RAG).

The process:
1. Load PDF documents and split them into chunks
2. Generate vector embeddings for each chunk using a local model
3. Store embeddings in a local Chroma database for fast retrieval
"""

import os
import glob
from dotenv import load_dotenv

# Load environment variables from .env file
# This reads configuration like where to store the vector database
load_dotenv()

from transformers import AutoTokenizer
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# Automatically find all PDF files in the source_documents directory
# This makes it easy to add/remove PDFs without changing code
SOURCE_DOCUMENTS_DIR = "source_documents"
SOURCE_DOCUMENTS = glob.glob(os.path.join(SOURCE_DOCUMENTS_DIR, "*.pdf"))

# Name of the collection in the vector database
COLLECTION_NAME = "doc_index"

# The embedding model to use - this runs locally on your computer
# This is a lightweight model from HuggingFace that converts text to vectors
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def main():
    """
    Main function to orchestrate document indexing.
    
    Steps:
    1. Ingest all PDF documents and convert to text chunks
    2. Generate embeddings and persist to local Chroma database
    """
    if not SOURCE_DOCUMENTS:
        print(f"No PDF files found in {SOURCE_DOCUMENTS_DIR}/")
        print("Please add PDF files to the source_documents folder and try again.")
        return
    
    print(f"Found {len(SOURCE_DOCUMENTS)} PDF file(s) to index:")
    for doc in SOURCE_DOCUMENTS:
        print(f"  - {doc}")
    
    print("\nIngesting documents...")
    all_docs = ingest_docs(SOURCE_DOCUMENTS)
    print(f"Processed {len(all_docs)} document chunks")
    
    print("Persisting embeddings to local Chroma database...")
    db = generate_embed_index(all_docs)
    print("Done! Vector database is ready for use.")


def ingest_docs(source_documents):
    """
    Load and process multiple PDF documents.
    
    Args:
        source_documents: List of paths to PDF files
        
    Returns:
        List of document chunks ready for embedding
    """
    all_docs = []
    for source_doc in source_documents:
        print(f"Processing: {source_doc}")
        docs = pdf_to_chunks(source_doc)
        all_docs = all_docs + docs
    return all_docs


def pdf_to_chunks(pdf_file):
    """
    Convert a PDF file into text chunks suitable for embedding.
    
    Why we chunk:
    - Embedding models have token limits (512 tokens for this model)
    - Smaller chunks provide more precise semantic matches
    - Chunks preserve context from the original document
    
    Args:
        pdf_file: Path to the PDF file
        
    Returns:
        List of document chunks with metadata (page numbers, source file)
    """
    # Use the tokenizer from the embedding model to ensure chunks fit
    # This prevents text from being truncated during embedding
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    
    # Configure the text splitter to create chunks that respect token limits
    # separators: try to split on paragraphs first, then lines, then words
    # chunk_size: maximum 512 tokens (model's limit)
    # chunk_overlap: 0 means no overlap between chunks
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        separators=["\n \n", "\n\n", "\n", " ", ""],
        chunk_size=512,
        chunk_overlap=0,
    )
    
    # Load the PDF and split it into chunks
    # PyPDFLoader automatically extracts text and preserves page numbers
    loader = PyPDFLoader(pdf_file)
    docs = loader.load_and_split(text_splitter)
    return docs


def generate_embed_index(docs):
    """
    Generate vector embeddings and store them in a local Chroma database.
    
    Vector embeddings are numerical representations of text that capture semantic meaning.
    Similar text will have similar vector embeddings, allowing for semantic search.
    
    Args:
        docs: List of document chunks to embed
        
    Returns:
        Chroma database instance
    """
    # Initialize the embedding model - this runs locally on your computer
    # The model converts text into 384-dimensional vectors
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Get the directory where we'll store the vector database locally
    chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIR")
    
    if not chroma_persist_dir:
        raise EnvironmentError(
            "CHROMA_PERSIST_DIR not found in .env file. "
            "Please copy .env.default to .env and configure it."
        )
    
    # Create the Chroma vector database with the embeddings
    # This creates a local database in the specified directory
    db = create_index_chroma(docs, embeddings, chroma_persist_dir)
    return db


def create_index_chroma(docs, embeddings, persist_dir):
    """
    Create a local Chroma vector database from documents.
    
    Chroma is a lightweight vector database that runs entirely on your computer.
    It stores document chunks along with their vector embeddings for fast retrieval.
    
    Args:
        docs: Document chunks to store
        embeddings: Embedding model to convert text to vectors
        persist_dir: Local directory to store the database
        
    Returns:
        Chroma database instance
    """
    # Create the database from documents
    # This automatically generates embeddings for all document chunks
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir,
    )
    
    # Persist the database to disk
    # This saves the embeddings so we don't have to regenerate them each time
    db.persist()
    return db


if __name__ == "__main__":
    main()
