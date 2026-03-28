import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from src.rag.pipeline import RAGPipeline

# Initialize
rag = RAGPipeline()

# Run the built-in test
rag.test_pipeline()