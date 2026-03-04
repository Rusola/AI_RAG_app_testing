# How to Setup RAG Project on a New Mac

This guide walks you through setting up this Retrieval-Augmented Generation (RAG) chatbot project on a fresh Mac from scratch.

## Prerequisites

Before starting, make sure you have:
- **Mac with macOS 10.14+**
- **Ollama** installed - Download from https://ollama.ai
- **Python 3.8+** installed - Download from https://python.org

## Step 1: Install Ollama and Pull a Model

### 1a. Install Ollama
1. Download Ollama from https://ollama.ai
2. Run the installer and follow the on-screen instructions
3. Once installed, Ollama will run automatically in the background

### 1b. Install a Language Model

Open Terminal and run one of these commands to download a model:

```bash
# Recommended for beginners (faster, good quality)
ollama pull mistral

# Or use llama2
ollama pull llama2

# Or use the latest llama3 (more capable but slower)
ollama pull llama3
```

Check what models you have installed:
```bash
ollama list
```

**Note:** Models are large (3-10GB). This may take 10-30 minutes depending on your internet speed.

## Step 2: Clone or Copy the Project

### Option A: Clone from GitHub (if available)
```bash
git clone https://github.com/harvard-hbs/rag-example.git
cd rag-example
```

### Option B: Copy the project folder
1. Copy the entire `rag-example` folder to your desired location
2. Navigate to it in Terminal:
```bash
cd /path/to/rag-example
```

## Step 3: Set Up Python Environment

### 3a. Create a Virtual Environment

```bash
# Create a virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate
```

You should see `(venv)` at the start of your Terminal prompt.

### 3b. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- `langchain` - Framework for RAG applications
- `streamlit` - Web UI framework
- `chromadb` - Local vector database
- `sentence-transformers` - For embeddings
- `pypdf` - For reading PDF files
- `python-dotenv` - For environment configuration

## Step 4: Configure Environment Variables

### 4a. Create `.env` File

```bash
# Copy the template to create your .env file
cp .env.default .env
```

### 4b. Edit `.env` File

Open `.env` in your text editor and configure it for your model:

```bash
# For mistral model:
OLLAMA_MODEL_NAME=mistral
OLLAMA_BASE_URL=http://localhost:11434
CHROMA_PERSIST_DIR=doc_index

# For llama2 model:
OLLAMA_MODEL_NAME=llama2
OLLAMA_BASE_URL=http://localhost:11434
CHROMA_PERSIST_DIR=doc_index

# For llama3 model:
OLLAMA_MODEL_NAME=llama3
OLLAMA_BASE_URL=http://localhost:11434
CHROMA_PERSIST_DIR=doc_index
```

**Important:** Make sure `OLLAMA_MODEL_NAME` matches a model you installed in Step 1!

## Step 5: Prepare Your Documents

### 5a. Add PDF Files

1. Create or locate your PDF documents
2. Place them in the `source_documents/` folder:
   ```bash
   mv /path/to/your/documents.pdf source_documents/
   ```

**Note:** The project comes with some example documents. You can replace them or add more.

### 5b. Index the Documents

This creates embeddings and stores them in the local database:

```bash
python index_documents.py
```

Expected output:
```
Loading documents from source_documents/
Creating embeddings... (may take 2-5 minutes)
Storing in Chroma database at doc_index/
✓ Indexing complete
```

The `doc_index/` folder will be created with the vector database.

## Step 6: Test the System

### 6a. Test Document Search (Command Line)

```bash
python search_index.py
```

Enter a question about your documents to verify the system works.

### 6b. Test Search UI (Web Browser)

```bash
streamlit run search_index_ui.py
```

- Opens automatically at http://localhost:8501
- Try searching for documents
- Press `Ctrl+C` to stop

## Step 7: Run the Chatbot

### Option A: Command-Line Chatbot

```bash
python document_chatbot.py
```

Enter questions and have a conversation with the AI about your documents.

### Option B: Web-Based Chatbot (Recommended)

```bash
streamlit run document_chatbot_ui.py
```

- Opens automatically at http://localhost:8501
- Beautiful web interface with chat history
- Shows source documents for each answer
- Press `Ctrl+C` in Terminal to stop

## Troubleshooting

### Problem: "Could not connect to Ollama"
**Solution:** Make sure Ollama is running
```bash
# Check if Ollama is running
ps aux | grep ollama

# If not running, start it
open /Applications/Ollama.app
```

### Problem: "Model not found" error
**Solution:** Install the model first
```bash
# Check installed models
ollama list

# Install the model you specified in .env
ollama pull mistral
```

### Problem: "No module named 'streamlit'"
**Solution:** Activate your virtual environment and reinstall requirements
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Problem: "Port 8501 already in use"
**Solution:** Streamlit is already running on another terminal. Either:
- Close the other Streamlit window, or
- Run on a different port:
```bash
streamlit run document_chatbot_ui.py --server.port=8502
```

### Problem: Long wait when starting Streamlit
**This is normal** - first run loads models and creates the Chroma database. Subsequent runs are faster.

## Project Structure

```
rag-example/
├── README.md                      # Original project documentation
├── README_2_SETUP_NEW_MAC.md     # This file
├── LOCAL_SETUP_GUIDE.md          # Detailed technical guide
├── requirements.txt              # Python dependencies
├── .env.default                  # Template for environment config
├── .env                          # Your configuration (create from .env.default)
│
├── document_chatbot.py           # Command-line chatbot
├── document_chatbot_ui.py        # Web UI chatbot (Streamlit)
├── search_index.py               # Command-line document search
├── search_index_ui.py            # Web UI document search (Streamlit)
├── index_documents.py            # Script to index your PDFs
│
├── source_documents/             # Place your PDF files here
├── doc_index/                    # Vector database (created by index_documents.py)
└── images/                       # Project images for documentation
```

## Quick Start Cheat Sheet

After initial setup, use these commands to run the project:

```bash
# Activate environment
source venv/bin/activate

# Make sure Ollama is running
open /Applications/Ollama.app

# Index documents (only needed after adding new PDFs)
python index_documents.py

# Run web chatbot
streamlit run document_chatbot_ui.py

# Or command-line chatbot
python document_chatbot.py
```

Then open http://localhost:8501 in your browser.

## Tips for Best Results

1. **Use a faster model if slow:** `mistral` is faster than `llama2`
2. **Use a better model if inaccurate:** `llama3` is better than `llama2`
3. **Keep PDFs small:** Break large documents into smaller files for better indexing
4. **Ask specific questions:** "What does document X say about Y?" works better than vague questions
5. **Check source documents:** Each answer shows which documents were used

## Need Help?

- **Ollama issues:** Visit https://github.com/ollama/ollama
- **Streamlit issues:** Visit https://discuss.streamlit.io
- **LangChain issues:** Visit https://github.com/langchain-ai/langchain
- **This project:** Check the LOCAL_SETUP_GUIDE.md for more technical details

## Next Steps

Once everything is working:
1. **Customize the system prompt** in `document_chatbot_ui.py` (around line 100)
2. **Add your own PDFs** to `source_documents/` and re-run `index_documents.py`
3. **Change the model** in `.env` to use a different LLM
4. **Adjust settings** like temperature and context window size

Enjoy your local RAG chatbot! 🚀
