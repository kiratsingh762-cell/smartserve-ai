"""
BUILD KNOWLEDGE BASE — Run this ONCE to:
  1. Load all cleaned documents from data/processed/
  2. Chunk them into ~600 char pieces
  3. Generate embeddings via OpenAI API
  4. Store everything in ChromaDB on disk

After this runs successfully, NEVER run it again unless
your source data changes. The vector store persists on disk.
"""
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from src.ingestion.data_loader import DataLoader
from src.knowledge_base.chunker import TextChunker
from src.knowledge_base.vectorstore import VectorStoreManager


def build():
    print("=" * 55)
    print("  SMARTSERVE AI -- KNOWLEDGE BASE BUILDER")
    print("=" * 55)

    # ── STEP 1: Load cleaned documents ───────────────────────
    print("\n[STEP 1/3] Loading processed documents...")

    # Try to load from saved CSV first (faster than re-processing)
    processed_path = "./data/processed/knowledge_base.csv"
    if os.path.exists(processed_path):
        import pandas as pd
        df = pd.read_csv(processed_path, encoding='utf-8')
        print(f"[OK] Loaded {len(df):,} documents from processed CSV")

        # Convert CSV rows back into document dicts
        raw_docs = []
        for _, row in df.iterrows():
            metadata = {k: str(v) for k, v in row.items()
                        if k not in ['text', 'source']}
            raw_docs.append({
                "text":     str(row['text']),
                "source":   str(row['source']),
                "metadata": metadata
            })
    else:
        # Fall back to full reload if CSV missing
        loader = DataLoader()
        raw_docs = loader.load_all()

    print(f"[OK] {len(raw_docs):,} documents ready for chunking")

    # ── STEP 2: Chunk documents ───────────────────────────────
    print("\n[STEP 2/3] Chunking documents...")
    chunker = TextChunker()
    chunks = chunker.chunk_documents(raw_docs)

    # Preview 3 sample chunks so you can verify quality
    chunker.preview_chunks(chunks, n=3)

    # ── STEP 3: Build vector store ────────────────────────────
    print("\n[STEP 3/3] Building vector store (calls OpenAI API)...")

    vs_manager = VectorStoreManager()

    # Safety check — don't rebuild if already exists
    if vs_manager.is_built():
        answer = input("\n[WARNING] Vector store already exists!\n"
                       "Rebuilding will cost API money.\n"
                       "Type 'yes' to rebuild, anything else to skip: ")
        if answer.strip().lower() != 'yes':
            print("[SKIP] Using existing vector store.")
            return

    vs_manager.build_vectorstore(chunks)

    # ── VERIFY ────────────────────────────────────────────────
    print("\n[VERIFY] Testing retrieval with a sample query...")
    vs_manager.load_vectorstore()
    results = vs_manager.similarity_search("my device is not turning on", k=2)

    print("\nTop 2 results for: 'my device is not turning on'")
    print("-" * 55)
    for i, (doc, score) in enumerate(results, 1):
        print(f"\nResult {i} (score: {score:.4f})")
        print(f"Source : {doc.metadata.get('source', 'N/A')}")
        print(f"Preview: {doc.page_content[:200]}...")

    print("\n" + "=" * 55)
    print("  KNOWLEDGE BASE BUILD COMPLETE!")
    print(f"  Location: {os.path.abspath('./data/vectordb')}")
    print("=" * 55)
    print("\nNext: Run the API and UI")
    print("  Terminal 1: uvicorn src.api.main:app --reload --port 8000")
    print("  Terminal 2: streamlit run ui/chatbot.py")


if __name__ == "__main__":
    build() 