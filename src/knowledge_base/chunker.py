import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP


class TextChunker:
    """
    Splits documents into smaller chunks for efficient retrieval.

    CONCEPT - RecursiveCharacterTextSplitter:
    It tries to split text at natural boundaries in this ORDER:
      1. Paragraph breaks  (two newlines: \\n\\n)
      2. Line breaks       (one newline:  \\n)
      3. Sentences         (period+space: ". ")
      4. Word boundaries   (space: " ")
      5. Last resort       (characters)

    It always tries the most natural split first.
    This preserves meaning — it never cuts a sentence in half
    if it can avoid it.

    chunk_size    = max characters per chunk (we use 600)
    chunk_overlap = how many characters chunks SHARE with neighbors (60)

    WHY OVERLAP?
    Imagine a document says:
      "...the warranty covers 12 months. For repairs, contact support..."
    If we cut exactly at "months." the next chunk starts at "For repairs"
    and loses the warranty context. Overlap means both chunks include
    "...12 months. For repairs..." so context is never lost at boundaries.
    """

    def __init__(self, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )

    def chunk_documents(self, raw_docs: list) -> list:
        """
        Converts raw document dicts into LangChain Document objects
        and splits them into chunks.

        INPUT:  list of {"text": "...", "source": "...", "metadata": {...}}
        OUTPUT: list of LangChain Document objects (chunks)

        CONCEPT - LangChain Document:
        LangChain has its own Document class with two fields:
          page_content = the text of this chunk
          metadata     = dict of extra info (source, ticket_id, etc.)
        We use this format because our VectorStore (ChromaDB) expects it.
        """
        # Convert dicts to LangChain Documents
        langchain_docs = []
        for doc in raw_docs:
            langchain_docs.append(
                Document(
                    page_content=doc["text"],
                    metadata={
                        "source": doc["source"],
                        **doc["metadata"]
                    }
                )
            )

        # Split all documents into chunks
        chunks = self.splitter.split_documents(langchain_docs)

        # Add chunk index to metadata for traceability
        # CONCEPT - Traceability: In production, you want to know
        # WHICH chunk of WHICH document gave the AI its answer.
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = str(i)

        print(f"[CHUNK] {len(langchain_docs):,} documents -> {len(chunks):,} chunks")
        print(f"[CHUNK] Avg chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")
        return chunks

    def preview_chunks(self, chunks: list, n: int = 3):
        """
        Shows a preview of n chunks for inspection.
        Use this to verify chunking quality before embedding.
        """
        print(f"\n--- CHUNK PREVIEW (showing {n} samples) ---")
        step = max(1, len(chunks) // n)
        for i in range(0, min(n * step, len(chunks)), step):
            chunk = chunks[i]
            print(f"\nChunk #{i}")
            print(f"Source  : {chunk.metadata.get('source', 'N/A')}")
            print(f"Length  : {len(chunk.page_content)} chars")
            print(f"Preview : {chunk.page_content[:150]}...")
            print("-" * 50)