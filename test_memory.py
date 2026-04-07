import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from src.rag.pipeline import RAGPipeline
rag = RAGPipeline()
rag.test_pipeline()