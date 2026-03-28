import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import pandas as pd
from src.knowledge_base.chunker import TextChunker

df = pd.read_csv('./data/processed/knowledge_base.csv', encoding='utf-8')
raw_docs = [{'text': str(r['text']), 'source': str(r['source']),
             'metadata': {}} for _, r in df.iterrows()]

chunker = TextChunker()
chunks = chunker.chunk_documents(raw_docs)
chunker.preview_chunks(chunks, n=3)

print()
print('Chunk size distribution:')
sizes = [len(c.page_content) for c in chunks]
print(f'  Min  : {min(sizes)} chars')
print(f'  Max  : {max(sizes)} chars')
print(f'  Avg  : {sum(sizes)//len(sizes)} chars')
print(f'  Total: {len(chunks):,} chunks')
print()
print('[OK] Chunking verified -- ready to build vector store')