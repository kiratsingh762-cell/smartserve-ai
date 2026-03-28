# test_setup.py — Run this to verify your entire setup is working
# Delete this file after testing

print("Testing SmartServe AI setup...\n")

# Test 1: Settings load correctly
print("1️⃣  Testing config...")
from config.settings import OPENAI_API_KEY, LLM_MODEL, CHROMA_PERSIST_DIR
print(f"   ✅ Model: {LLM_MODEL}")
print(f"   ✅ Vector DB path: {CHROMA_PERSIST_DIR}")

# Test 2: OpenAI connection
print("\n2️⃣  Testing OpenAI connection...")
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Say: SmartServe AI is ready!"}],
    max_tokens=20
)
print(f"   ✅ OpenAI response: {response.choices[0].message.content}")

# Test 3: LangChain import
print("\n3️⃣  Testing LangChain...")
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
print("   ✅ LangChain imported successfully")

# Test 4: ChromaDB
print("\n4️⃣  Testing ChromaDB...")
import chromadb
client_chroma = chromadb.Client()
collection = client_chroma.create_collection("test_collection")
collection.add(documents=["Hello SmartServe"], ids=["1"])
result = collection.query(query_texts=["Hello"], n_results=1)
print(f"   ✅ ChromaDB working: {result['documents'][0][0]}")
client_chroma.delete_collection("test_collection")

# Test 5: Pandas
print("\n5️⃣  Testing Pandas...")
import pandas as pd
df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
print(f"   ✅ Pandas working | DataFrame shape: {df.shape}")

print("\n" + "="*45)
print("🎉  ALL TESTS PASSED! Your setup is complete.")
print("="*45)
print("\nNext step: Module 3 — Download Kaggle Datasets")
