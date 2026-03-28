# test_data_loader.py
# Fix Windows cp1252 encoding issue — must be FIRST line of code
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.ingestion.data_loader import DataLoader

# Initialize and run
loader = DataLoader()
documents = loader.load_all()
# ... rest of your file stays exactly the same

# Inspect a sample from each source
print("\n" + "="*55)
print("SAMPLE DOCUMENT INSPECTION")
print("="*55)

sources = ['support_tickets', 'faq', 'ecommerce_intents']
for source in sources:
    sample = next((d for d in documents if d['source'] == source), None)
    if sample:
        print(f"\n📄 SOURCE: {source.upper()}")
        print(f"METADATA: {sample['metadata']}")
        print(f"TEXT PREVIEW (first 300 chars):")
        print(f"{sample['text'][:300]}...")
        print("-"*55)

# Save to processed/
loader.save_processed(documents)
print("\n✅ Data loader test complete!")

import pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
df = pd.read_csv('./data/processed/knowledge_base.csv', encoding='utf-8')
double = df['text'].str.contains('[{][{]', regex=True, na=False).sum()
print(f'Total docs    : {len(df):,}')
print(f'Double braces : {double}')
print('CLEAN [OK] -- READY FOR MODULE 5' if double == 0 else 'DIRTY -- paste to mentor')