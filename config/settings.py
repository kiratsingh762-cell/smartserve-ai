import os
from dotenv import load_dotenv

# Load the .env file — must be called before accessing os.getenv()
load_dotenv()

# ─── OpenAI Settings ───────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"   # For converting text → vectors
LLM_MODEL = "gpt-4o"                         # The AI brain for answering
LLM_TEMPERATURE = 0                          # 0 = consistent/factual answers

# ─── Vector Database Settings ──────────────────────────────
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/vectordb")
COLLECTION_NAME = "smartserve_knowledge"

# ─── Chunking Settings ─────────────────────────────────────
CHUNK_SIZE = 500       # Each chunk = max 500 characters
CHUNK_OVERLAP = 50     # Chunks share 50 chars with neighbors for context

# ─── Retrieval Settings ────────────────────────────────────
TOP_K_RESULTS = 4      # Fetch top 4 most relevant chunks per query

# ─── AS400 Settings ────────────────────────────────────────
AS400_DB_PATH = os.getenv("AS400_SIMULATED_DB", "./data/simulated_as400/orders.csv")

# ─── Data Paths ────────────────────────────────────────────
RAW_DATA_PATH = "./data/raw"
PROCESSED_DATA_PATH = "./data/processed"

# ─── Validation ────────────────────────────────────────────
# This runs when settings.py is imported — catches missing keys early
if not OPENAI_API_KEY:
    raise ValueError(
        "❌ OPENAI_API_KEY is missing! "
        "Add it to your .env file: OPENAI_API_KEY=sk-proj-..."
    )

print(f"✅ Settings loaded | Model: {LLM_MODEL} | Embedding: {EMBEDDING_MODEL}")
