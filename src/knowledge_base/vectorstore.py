import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from config.settings import (
    OPENAI_API_KEY, EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR, COLLECTION_NAME, TOP_K_RESULTS
)


class VectorStoreManager:
    """
    Manages the ChromaDB vector database.

    CONCEPT - What ChromaDB stores for each chunk:
    +-----------+----------------------------------+------------------+
    | chunk_id  | page_content (text)              | embedding vector |
    +-----------+----------------------------------+------------------+
    |     0     | "Product: Dell XPS Issue Type..."| [0.02,-0.41,...] |
    |     1     | "...not charging. Resolution:..."| [0.05,-0.38,...] |
    |     2     | "Category: ORDER Intent:cancel.."| [0.11, 0.22,...] |
    +-----------+----------------------------------+------------------+

    When a customer asks a question:
    1. Question is converted to a vector
    2. ChromaDB finds the TOP_K_RESULTS closest vectors
    3. Those chunks are returned as context for the LLM

    CONCEPT - Cosine Similarity (how ChromaDB ranks results):
    Measures the angle between two vectors.
    Angle = 0   (same direction) → similarity = 1.0  (identical meaning)
    Angle = 90  (perpendicular)  → similarity = 0.0  (unrelated)
    Angle = 180 (opposite)       → similarity = -1.0 (opposite meaning)
    ChromaDB returns chunks with similarity CLOSEST to 1.0
    """

    def __init__(self):
        # OpenAIEmbeddings converts text → 1,536-dimensional vectors
        # text-embedding-3-small: fast, cheap, accurate enough for support
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY
        )
        self.vectorstore = None
        print(f"[OK] Embedding model loaded: {EMBEDDING_MODEL}")

    def build_vectorstore(self, chunks: list):
        """
        Converts all chunks to embeddings and stores in ChromaDB.

        CONCEPT - This is the EXPENSIVE step (in time and API cost):
        For each of ~22,000 chunks, we call OpenAI's API to get a
        1,536-number vector. OpenAI charges per 1,000 tokens processed.
        For our dataset: ~$0.50-1.50 total. Done ONCE, then reused forever.

        Chroma.from_documents() does three things internally:
          1. Calls OpenAI embeddings API for every chunk
          2. Stores the text + metadata in ChromaDB
          3. Stores the embedding vectors in ChromaDB
          4. Persists everything to disk at CHROMA_PERSIST_DIR
        """
        print(f"\n[EMBED] Building vector store...")
        print(f"[EMBED] Chunks to embed: {len(chunks):,}")
        print(f"[EMBED] This calls OpenAI API -- may take 5-15 minutes...")
        print(f"[EMBED] Cost estimate: ~$0.50-1.50 USD (one-time)")

        # Process in batches to avoid API rate limits
        # CONCEPT - Rate Limits: OpenAI allows max N API calls per minute.
        # If we send all 22,000 chunks at once, it throttles us.
        # ChromaDB handles batching internally, but we set batch_size
        # as a safety measure.
        BATCH_SIZE = 500
        total_batches = (len(chunks) // BATCH_SIZE) + 1

        print(f"[EMBED] Processing in {total_batches} batches of {BATCH_SIZE}...")

        # First batch — creates the collection
        first_batch = chunks[:BATCH_SIZE]
        self.vectorstore = Chroma.from_documents(
            documents=first_batch,
            embedding=self.embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
            collection_name=COLLECTION_NAME
        )

        # Remaining batches — add to existing collection
        for i in range(1, total_batches):
            start = i * BATCH_SIZE
            end   = min(start + BATCH_SIZE, len(chunks))
            batch = chunks[start:end]

            if not batch:
                break

            self.vectorstore.add_documents(batch)
            progress = ((i + 1) / total_batches) * 100
            print(f"   Batch {i+1}/{total_batches} done ({progress:.0f}%)")

        # Chroma 0.4+ auto-persists -- no manual call needed
        print(f"\n[OK] Vector store built and saved to: {CHROMA_PERSIST_DIR}")
        print(f"[OK] Total vectors stored: {len(chunks):,}")
        return self.vectorstore

    def load_vectorstore(self):
        """
        Loads an existing vector store from disk.
        Use this every time AFTER the first build — no API calls needed.

        CONCEPT - Persistence:
        ChromaDB saves everything to disk in CHROMA_PERSIST_DIR.
        Loading from disk is instant — no re-embedding, no API cost.
        This is why we build once and reuse forever (until data changes).
        """
        print(f"[OK] Loading vector store from: {CHROMA_PERSIST_DIR}")
        self.vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=self.embeddings,
            collection_name=COLLECTION_NAME
        )
        count = self.vectorstore._collection.count()
        print(f"[OK] Vectors in store: {count:,}")
        return self.vectorstore

    def similarity_search(self, query: str, k: int = TOP_K_RESULTS):
        """
        Finds the top-k most relevant chunks for a query.

        CONCEPT - What happens here step by step:
        1. query text → OpenAI API → query vector (1,536 numbers)
        2. ChromaDB compares query vector against ALL stored vectors
        3. Returns top-k chunks with highest cosine similarity scores
        4. Score close to 0 = very relevant (small distance)
           Score close to 2 = not relevant (large distance)

        Note: ChromaDB uses L2 distance by default (lower = better match)
        """
        if not self.vectorstore:
            self.load_vectorstore()

        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return results

    def is_built(self) -> bool:
        """Check if vector store already exists on disk."""
        db_path = Path(CHROMA_PERSIST_DIR)
        return db_path.exists() and any(db_path.iterdir())