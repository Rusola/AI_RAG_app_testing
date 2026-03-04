"""
The simplest script for embedding-based retrieval.

This script demonstrates how to search the vector database using semantic similarity.
Instead of keyword matching, it finds documents whose meaning is similar to your query.

How it works:
1. Load the local Chroma vector database
2. Convert your query into a vector embedding
3. Find document chunks with similar embeddings
4. Display the most relevant chunks with similarity scores
"""

import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# Collection name must match the one used during indexing
COLLECTION_NAME = "doc_index"

# Embedding model must be the same one used to create the index
# Using a different model would produce incompatible vectors
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def main():
    """
    Perform a semantic search on the indexed documents.
    
    This demonstrates how vector similarity search works:
    - The query is converted to a vector embedding
    - The database finds chunks with similar vector embeddings
    - Lower similarity scores indicate better matches
    """
    # Initialize the same embedding model used during indexing
    # This ensures query embeddings are compatible with stored document embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Connect to the local Chroma vector database
    db = get_embed_db(embeddings)

    # Example query to search for similar documents
    # The search uses semantic meaning, not just keyword matching
    prompt = (
        "How should government responsibility be divided between "
        "the states and the federal government?"
    )

    # Perform similarity search and get results with scores
    # Lower scores = better matches (closer vectors in embedding space)
    print(f"Finding document matches for '{prompt}'")
    docs_scores = db.similarity_search_with_score(prompt)
    
    # Display the results
    for doc, score in docs_scores:
        print(f"\nSimilarity score (lower is better): {score}")
        print(doc.metadata)  # Shows source file and page number
        print(doc.page_content)  # Shows the actual text chunk


def get_embed_db(embeddings):
    """
    Load the local Chroma vector database.
    
    This connects to the persisted database created by index_documents.py.
    All operations happen locally on your computer - no cloud services.
    
    Args:
        embeddings: The embedding model instance
        
    Returns:
        Chroma database instance ready for searching
    """
    # Get the database directory from environment variables
    chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIR")
    
    if not chroma_persist_dir:
        raise EnvironmentError(
            "CHROMA_PERSIST_DIR not found in .env file. "
            "Please copy .env.default to .env and configure it."
        )
    
    # Connect to the local Chroma database
    db = get_chroma_db(embeddings, chroma_persist_dir)
    return db


def get_chroma_db(embeddings, persist_dir):
    """
    Create a connection to the persisted Chroma database.
    
    Args:
        embeddings: Embedding model instance
        persist_dir: Path to the local database directory
        
    Returns:
        Chroma database instance
    """
    db = Chroma(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir,
    )
    return db


if __name__ == "__main__":
    main()
