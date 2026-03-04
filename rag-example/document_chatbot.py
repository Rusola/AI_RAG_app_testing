"""
Simplest script for creating retrieval pipeline and invoking an LLM.

This demonstrates the full RAG (Retrieval-Augmented Generation) pipeline:
1. Retrieve relevant document chunks from the vector database based on your question
2. Pass those chunks to the local Ollama LLM as context
3. Generate an answer that's grounded in the retrieved documents

The LLM runs locally on your Mac via Ollama, and the vector database is also local.
No data is sent to cloud services.
"""

import os
import pprint
from dotenv import load_dotenv

# Load environment variables from the .env file
# This reads settings like the Ollama model name and database location
load_dotenv()

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOllama
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory

# Set to True to see detailed logs of what's sent to the LLM
VERBOSE = False

# Collection name must match the one used during indexing
COLLECTION_NAME = "doc_index"

# Embedding model must be the same one used to create the index
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Number of previous conversation turns to remember
# Higher = more context, but uses more memory and tokens
MEMORY_WINDOW_SIZE = 10


def main():
    """
    Main function demonstrating the full RAG pipeline.
    
    This sets up:
    1. Connection to the local vector database
    2. Connection to the local Ollama LLM
    3. Conversation memory to maintain context
    4. A retrieval chain that combines everything
    """
    # Get Ollama configuration from environment variables
    ollama_model_name = os.getenv("OLLAMA_MODEL_NAME")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    if not ollama_model_name:
        raise EnvironmentError(
            "OLLAMA_MODEL_NAME not found in .env file. "
            "Please copy .env.default to .env and configure it. "
            "Make sure you have Ollama running and a model installed."
        )

    # Initialize the embedding model (runs locally)
    # This is used to convert queries into vectors for database search
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Connect to the local Chroma vector database
    # This contains the embeddings of your documents
    db = get_embed_db(embeddings)
    
    # Create a retriever that will fetch relevant documents
    # The retriever searches the database for chunks similar to the query
    retriever = db.as_retriever()

    # Initialize the Ollama LLM (runs locally on your Mac)
    # temperature controls randomness: 0 = deterministic, 1 = creative
    print(f"Using Ollama with model: {ollama_model_name}")
    print(f"Connecting to Ollama at: {ollama_base_url}")
    llm = ChatOllama(
        base_url=ollama_base_url,
        model=ollama_model_name,
        temperature=0.3,  # Lower temperature = more focused/concise,
        num_predict=75,  # Limits response to ~75 tokens (~300 characters),
    )

    # Set up conversation memory
    # This allows the chatbot to remember previous exchanges in the conversation
    # memory_key: the key used to pass chat history to the chain
    # output_key: which part of the response to store in memory
    # return_messages: return as message objects rather than strings
    # window_size: how many conversation turns to remember
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
        window_size=MEMORY_WINDOW_SIZE,
    )

    # Create the full RAG chain
    # This combines the LLM, memory, and retriever into one pipeline
    # When you ask a question:
    # 1. The retriever finds relevant document chunks
    # 2. Those chunks + your question + chat history are sent to the LLM
    # 3. The LLM generates an answer based on the retrieved context
    # 4. The exchange is stored in memory for future reference
    query_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=retriever,
        verbose=VERBOSE,
        return_source_documents=True,  # Include source documents in the response
    )

    # Example query to test the system
    prompt = (
        "How should government responsibility be divided between "
        "the states and the federal government?"
    )
    
    print(f"\nQuery: {prompt}\n")
    
    # Invoke the chain with the question
    # The response includes the answer and the source documents used
    query_response = query_chain({"question": prompt})
    
    # Display the full response
    pprint.pprint(query_response)


def get_embed_db(embeddings):
    """
    Load the local Chroma vector database.
    
    This connects to the persisted database created by index_documents.py.
    All operations happen locally - no cloud services.
    
    Args:
        embeddings: The embedding model instance
        
    Returns:
        Chroma database instance
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
