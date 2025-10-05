import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- API Keys ---
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Set LangSmith tracing environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangGraph RAG Weather Agent" # Optional: a name for your project in LangSmith
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY


# --- LLM and Embedding Models ---
LLM_MODEL_NAME = "llama-3.3-70b-versatile"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# The vector size for the embedding model above.
# For "text-embedding-3-small", it's 1536.
# For "text-embedding-3-large", it's 3072.
EMBEDDING_MODEL_DIMENSION = 384

# --- Data ---
PDF_PATH = "data/sample.pdf"

# --- Vector Database (Qdrant) ---
# Use a local, file-based Qdrant instance
QDRANT_STORAGE_PATH = os.getenv("QDRANT_STORAGE_PATH", "data/qdrant_storage")
QDRANT_COLLECTION_NAME = "pdf_document_collection"
