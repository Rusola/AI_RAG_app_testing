"""
User interface for seeing vector database matches.

This Streamlit app provides an interactive web interface for searching the vector database.
You can enter queries and see which document chunks match your search, along with
similarity scores and source information.

Usage: streamlit run search_index_ui.py
"""

import os
import streamlit as st
import pprint
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

from streamlit.logger import get_logger

logger = get_logger(__name__)

# Collection name must match the one used during indexing
COLLECTION_NAME = "doc_index"

# Embedding model must be the same one used to create the index
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# UI configuration
ANSWER_ROLE = "Document Index"
FIRST_MESSAGE = "Enter text to find document matches."
QUESTION_ROLE = "Searcher"
PLACE_HOLDER = "Your message"


# Cached shared objects - these are loaded once and reused across the session
# This improves performance by not reloading the model and database for each query

@st.cache_resource
def load_embeddings():
    """
    Load the embedding model once and cache it.
    
    The cache prevents reloading the model on every interaction.
    multi_process=False is required for Streamlit compatibility.
    
    Returns:
        HuggingFace embedding model instance
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, multi_process=False)
    return embeddings


@st.cache_resource
def get_embed_db():
    """
    Connect to the local Chroma vector database and cache the connection.
    
    This loads the persisted database created by index_documents.py.
    The cache ensures we don't reconnect on every query.
    
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
    
    # Connect to the local database
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


# Initialize the shared resources
# These will be cached and reused across all user sessions
embeddings = load_embeddings()
db = get_embed_db()


def save_message(role, content, sources=None):
    """
    Save a message to the session state for display.
    
    Args:
        role: Who sent the message (user or system)
        content: The message text
        sources: Optional list of source documents
        
    Returns:
        The message dictionary that was saved
    """
    logger.info(f"message: {role} - '{content}'")
    msg = {"role": role, "content": content, "sources": sources}
    st.session_state["messages"].append(msg)
    return msg


def write_message(msg):
    """
    Display a message in the chat interface.
    
    Shows the message content and any associated source documents
    with their metadata and text content.
    
    Args:
        msg: Message dictionary with role, content, and optional sources
    """
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["sources"]:
            for doc in msg["sources"]:
                # Display metadata (source file, page number, similarity score)
                st.text(pprint.pformat(doc.metadata))
                # Display the actual text content from the document
                st.write(doc.page_content)


# Streamlit UI setup
st.title("Show Document Matches")

# Initialize message history in session state
# This persists messages across interactions within a session
if "messages" not in st.session_state:
    st.session_state["messages"] = []
    save_message(ANSWER_ROLE, FIRST_MESSAGE)

# Display all previous messages
for msg in st.session_state["messages"]:
    write_message(msg)

# Handle new user input
if prompt := st.chat_input(PLACE_HOLDER):
    # Save and display the user's query
    msg = save_message(QUESTION_ROLE, prompt)
    write_message(msg)

    # Perform similarity search on the vector database
    # This converts the query to a vector and finds similar document vectors
    docs_scores = db.similarity_search_with_score(prompt)
    
    # Add similarity scores to document metadata
    docs = []
    for doc, score in docs_scores:
        doc.metadata["similarity_score"] = score
        docs.append(doc)

    # Display the matching documents
    msg = save_message(ANSWER_ROLE, "Matching Documents", docs)
    write_message(msg)
