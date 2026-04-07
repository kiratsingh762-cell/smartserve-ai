import sys
import os
sys.path.insert(0, os.getcwd())

print("=" * 55)
print("  SmartServe API Diagnostics")
print("=" * 55)

steps = []

# 1. Settings
try:
    from config.settings import OPENAI_API_KEY, LLM_MODEL
    key_preview = OPENAI_API_KEY[:8] + "..." if OPENAI_API_KEY else "MISSING"
    steps.append(("[OK]", f"Settings loaded | Key: {key_preview} | Model: {LLM_MODEL}"))
except Exception as e:
    steps.append(("[FAIL]", f"Settings: {e}"))

# 2. Memory
try:
    from src.memory.session_memory import SessionMemoryManager
    steps.append(("[OK]", "SessionMemoryManager imported"))
except Exception as e:
    steps.append(("[FAIL]", f"SessionMemoryManager: {e}"))

# 3. VectorStore
try:
    from src.knowledge_base.vectorstore import VectorStoreManager
    steps.append(("[OK]", "VectorStoreManager imported"))
except Exception as e:
    steps.append(("[FAIL]", f"VectorStoreManager: {e}"))

# 4. Pipeline import
try:
    from src.rag.pipeline import RAGPipeline
    steps.append(("[OK]", "RAGPipeline imported"))
except Exception as e:
    steps.append(("[FAIL]", f"RAGPipeline import: {e}"))

# 5. Pipeline init
try:
    rag = RAGPipeline()
    steps.append(("[OK]", f"RAGPipeline initialized"))
except Exception as e:
    steps.append(("[FAIL]", f"RAGPipeline init: {e}"))

# 6. AS400
try:
    from src.as400.connector import AS400Connector
    as400 = AS400Connector()
    steps.append(("[OK]", "AS400Connector initialized"))
except Exception as e:
    steps.append(("[FAIL]", f"AS400Connector: {e}"))

# 7. Analytics
try:
    from src.analytics.logger import AnalyticsLogger
    analytics = AnalyticsLogger()
    steps.append(("[OK]", "AnalyticsLogger initialized"))
except Exception as e:
    steps.append(("[FAIL]", f"AnalyticsLogger: {e}"))

# 8. FastAPI
try:
    from src.api.main import app
    steps.append(("[OK]", "FastAPI app imported successfully"))
except Exception as e:
    steps.append(("[FAIL]", f"FastAPI app import: {e}"))

print()
for status, msg in steps:
    print(f"  {status}  {msg}")

print()
fails = [s for s in steps if s[0] == "[FAIL]"]
if fails:
    print(f"  RESULT: {len(fails)} issue(s) found — fix the [FAIL] items above")
else:
    print("  RESULT: All checks passed — API should start cleanly")
print("=" * 55)
