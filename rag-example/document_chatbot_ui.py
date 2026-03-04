"""
Interactive UI running against a local language model with
retrieval-augmented generation and memory.

This provides a web-based chat interface where you can:
- Ask questions about your indexed documents
- Get AI-generated answers grounded in the document content
- Maintain conversational context across multiple exchanges
- See which source documents were used for each answer

Everything runs locally on your Mac - no cloud services required.

Usage: streamlit run document_chatbot_ui.py

This will open your browser to the Streamlit UI.
Press Ctrl-C in the terminal to stop the server.

Prerequisites:
* Install the python requirements
* Setup .env file from .env.default
* Make sure Ollama is running with a model installed
* Run index_documents.py to create the document index
"""

import os

import streamlit as st
from dotenv import load_dotenv

# Load environment variables from the .env file
# This reads settings like the Ollama model name and database location
load_dotenv()

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOllama
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory

from streamlit.logger import get_logger

logger = get_logger(__name__)

# Set to True to see detailed logs of what's sent to the LLM
VERBOSE = False

# Collection name must match the one used during indexing
COLLECTION_NAME = "doc_index"

# Embedding model must be the same one used to create the index
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Number of previous conversation turns to remember
# Higher = more context, but uses more memory and tokens
MEMORY_WINDOW_SIZE = 10

# UI configuration
ANSWER_ROLE = "Chatbot"
FIRST_MESSAGE = "Hello my friend! How can I help you?"
QUESTION_ROLE = "User"
PLACE_HOLDER = "Your message"


# Cached shared objects - these are loaded once and reused across all sessions
# Caching prevents reloading models and reconnecting to databases on every interaction

@st.cache_resource
def load_embeddings():
    """
    Load the embedding model once and cache it.
    
    This model converts text into numerical vectors (embeddings).
    The cache prevents reloading the model on every interaction.
    
    Returns:
        HuggingFace embedding model instance
    """
    embeds = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return embeds


@st.cache_resource
def load_llm():
    """
    Initialize the local Ollama LLM and cache it.
    
    Ollama runs completely locally on your Mac - no data is sent to cloud services.
    Make sure Ollama.app is running and you have a model installed.
    
    To check available models: ollama list
    To install a model: ollama pull llama2
    
    Returns:
        ChatOllama instance configured from environment variables
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
    
    print(f"Using Ollama with model: {ollama_model_name}")
    print(f"Connecting to Ollama at: {ollama_base_url}")
    
    # Initialize the Ollama LLM
    # temperature: 0.5 provides a balance between consistent and creative responses
    # verbose: shows detailed logs if enabled
    llm = ChatOllama(
        base_url=ollama_base_url,
        model=ollama_model_name,
        temperature=0.3,  # Lower temperature = more focused/concise
        num_predict=500,  # Allows for longer responses in any language
        system="""You are a helpful, accurate, and concise AI assistant. Follow these guidelines:

        1. LANGUAGE: When asked to respond in a specific language, provide your ENTIRE answer ONLY in that language. Never mix languages."

        # 2. ACCURACY: Base your answers strictly on the provided document context. If the information isn't in the documents, clearly state "I don't find this information in the provided documents."

        # 3. BREVITY: Be concise and direct. Answer the question without unnecessary elaboration unless specifically asked for details.

        # 4. STRUCTURE: For complex answers, use clear formatting with bullet points or numbered lists when appropriate.

        # 5. CITATIONS: When referencing information, acknowledge it comes from the source documents.

        # 6. TONE: Be friendly and professional. Maintain a helpful demeanor while staying factual.

        # 7. NO SPECULATION: Never guess or speculate. Only state what's explicitly in the documents.

        # 8. QUESTIONS: If a question is ambiguous, ask for clarification rather than assuming.

        # 9. UPDATES: If document information seems outdated or contradictory, mention this.""",
    )
    return llm


@st.cache_resource
def get_embed_retriever():
    """
    Create a retriever for the vector database and cache it.
    
    A retriever is a LangChain interface for searching the vector database.
    It finds document chunks that are semantically similar to a query.
    
    Returns:
        Retriever instance for the Chroma database
    """
    db = get_embed_db(embeddings)
    # as_retriever() creates a standard interface for the chain to use
    retriever = db.as_retriever()
    return retriever


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
    
    Chroma is a lightweight vector database that runs locally.
    It stores document embeddings for semantic search.
    
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


# Initialize shared resources
# These are cached and reused across all user sessions for efficiency
embeddings = load_embeddings()  # HuggingFace embedding model (local)
llm = load_llm()  # Ollama LLM (local)
retriever = get_embed_retriever()  # Chroma database retriever (local)


def save_message(role, content, sources=None):
    """
    Save a message to the session state for display.
    
    This maintains the chat history in Streamlit's session state,
    which persists across interactions within a user session.
    
    Args:
        role: Who sent the message (user or chatbot)
        content: The message text
        sources: Optional list of source documents with metadata
        
    Returns:
        The message dictionary that was saved
    """
    logger.info(f"message: {role} - '{content}'")
    msg = {"role": role, "content": content, "sources": sources}
    st.session_state["messages"].append(msg)
    return msg


def source_description(md):
    """
    Format source document metadata for display.
    
    Shows the filename and page number where the information was found.
    This helps users verify and reference the source material.
    
    Args:
        md: Metadata dictionary with 'source' and 'page' keys
        
    Returns:
        Formatted string describing the source
    """
    descr = f"{md['source']}, Page {md['page']}"
    return descr


def write_message(msg):
    """
    Display a message in the chat interface.
    
    Shows the message content and any source documents that were used
    to generate the response. This provides transparency about where
    the AI got its information.
    
    Args:
        msg: Message dictionary with role, content, and optional sources
    """
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        # Display source documents if available
        if msg["sources"]:
            st.write(", ".join([source_description(md) for md in msg["sources"]]))


# Streamlit UI setup
st.title("Document Chatbot")

st.write(
    """This conversational interface allows you to interact with
indexed content, in this case, The Federalist Papers. The AI runs locally
on your Mac via Ollama - no data is sent to cloud services."""
)

# Initialize the conversational retrieval chain
# This combines the LLM, memory, and retriever into a single pipeline
if "query_chain" not in st.session_state:
    # Create conversation memory to maintain context across exchanges
    # This remembers the last MEMORY_WINDOW_SIZE conversation turns
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",  # Key used to pass history to the chain
        output_key="answer",  # Which part of the response to store
        return_messages=True,  # Return as message objects
        window_size=MEMORY_WINDOW_SIZE,  # How many turns to remember
    )
    
    # Create the full RAG chain
    # When you ask a question:
    # 1. The retriever finds relevant document chunks from the vector database
    # 2. Those chunks + your question + chat history are sent to Ollama
    # 3. Ollama generates an answer based on the retrieved context
    # 4. The exchange is stored in memory for future reference
    st.session_state["query_chain"] = ConversationalRetrievalChain.from_llm(
        llm=llm,  # Local Ollama LLM
        memory=memory,  # Conversation history
        retriever=retriever,  # Vector database retriever
        verbose=VERBOSE,  # Enable detailed logging if True
        return_source_documents=True,  # Include source docs in response
    )

# Initialize message history in session state
# This persists messages across interactions within a session
if "messages" not in st.session_state:
    st.session_state["messages"] = []
    save_message(ANSWER_ROLE, FIRST_MESSAGE)

# Display all previous messages in the conversation
for msg in st.session_state["messages"]:
    write_message(msg)

# Handle new user input
if prompt := st.chat_input(PLACE_HOLDER):
    # Save and display the user's question
    msg = save_message(QUESTION_ROLE, prompt)
    write_message(msg)

    # Show a loading message while generating the response
    with st.spinner("Please wait. I am thinking..."):
        # Invoke the RAG chain with the user's question
        # This will:
        # 1. Search the vector database for relevant chunks
        # 2. Send the chunks + question to Ollama
        # 3. Get an AI-generated response
        qa = st.session_state["query_chain"]
        query_response = qa({"question": prompt})
    
    # Extract the answer and source documents from the response
    response = query_response["answer"]
    source_docs = [d.metadata for d in query_response["source_documents"]]
    
    # Save and display the chatbot's response with sources
    msg = save_message(ANSWER_ROLE, response, source_docs)
    write_message(msg)
