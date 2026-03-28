import pandas as pd
import json
import os
import re
from pathlib import Path
import sys
from collections import Counter

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import RAW_DATA_PATH, PROCESSED_DATA_PATH


class DataLoader:

    TICKETS_FILE = "customer_support_tickets.csv"
    BITEXT_FILE  = "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
    INTENTS_FILE = "Ecommerce_FAQ_intents.json"

    def __init__(self, raw_path=RAW_DATA_PATH):
        self.raw_path = raw_path
        self._verify_files()

    def _verify_files(self):
        for f in [self.TICKETS_FILE, self.BITEXT_FILE, self.INTENTS_FILE]:
            full = os.path.join(self.raw_path, f)
            if not os.path.exists(full):
                raise FileNotFoundError(f"File not found: {full}")
        print("[OK] All dataset files found.")

    @staticmethod
    def _clean_text(text):
        if pd.isna(text) or text is None:
            return ""
        text = str(text).strip()
        # Remove closed double braces {{anything}}
        text = re.sub(r"[{][{][^}]*[}][}]", "", text)
        # Remove unclosed double braces {{anything (no closing))
        text = re.sub(r"[{][{][^}]*", "", text)
        # Remove closed single braces {anything}
        text = re.sub(r"[{][^}]*[}]", "", text)
        # Remove unclosed single braces {anything at end
        text = re.sub(r"[{][^}]*$", "", text)
        # Remove version numbers like 1.8.3
        text = re.sub(r"\b\d+\.\d+\.\d+\b", "", text)
        # Remove leftover dashes from removals
        text = re.sub(r"\s+-\s+", " ", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove non-ASCII characters
        text = re.sub(r'[^\x20-\x7E\n]', '', text)
        # Collapse repeated punctuation
        text = re.sub(r"([!?.])[!?.]+", r"\1", text)
        return text.strip()

    @staticmethod
    def _is_meaningful(text, min_words=8):
        if not text or not text.strip():
            return False
        words = [w for w in text.split() if len(w) > 1]
        if len(words) < min_words:
            return False
        nums = [w for w in words if w.replace(".", "").replace(",", "").isdigit()]
        if words and len(nums) > len(words) * 0.6:
            return False
        return True

    def load_support_tickets(self):
        print("[+] Loading Dataset 1: Customer Support Tickets...")
        df = pd.read_csv(
            os.path.join(self.raw_path, self.TICKETS_FILE),
            encoding="utf-8", on_bad_lines="skip"
        )
        print(f"    Raw rows: {len(df):,}")
        df.columns = df.columns.str.strip()
        df = df.dropna(subset=["Ticket Description", "Resolution"])
        df = df.drop_duplicates(subset=["Ticket Description"])
        print(f"    After dedup: {len(df):,}")
        for col in [
            "Ticket Description", "Resolution",
            "Ticket Type", "Product Purchased", "Ticket Priority"
        ]:
            if col in df.columns:
                df[col] = df[col].apply(DataLoader._clean_text)
        mask = (
            df["Ticket Description"].apply(DataLoader._is_meaningful) &
            df["Resolution"].apply(lambda x: DataLoader._is_meaningful(x, 5))
        )
        df = df[mask]
        print(f"    After quality filter: {len(df):,}")
        docs = []
        for _, row in df.iterrows():
            text = (
                f"Product: {row.get('Product Purchased', 'N/A')}\n"
                f"Issue Type: {row.get('Ticket Type', 'General')}\n"
                f"Priority: {row.get('Ticket Priority', 'Medium')}\n"
                f"Customer Problem: {row.get('Ticket Description', '')}\n"
                f"Proven Resolution: {row.get('Resolution', '')}"
            )
            docs.append({
                "text": text,
                "source": "support_tickets",
                "metadata": {
                    "ticket_id":   str(row.get("Ticket ID", "N/A")),
                    "product":     str(row.get("Product Purchased", "N/A")),
                    "ticket_type": str(row.get("Ticket Type", "N/A")),
                    "priority":    str(row.get("Ticket Priority", "N/A"))
                }
            })
        print(f"    [OK] Produced {len(docs):,} ticket documents")
        return docs

    def load_faq_data(self):
        print("[+] Loading Dataset 2: Bitext FAQ...")
        df = pd.read_csv(
            os.path.join(self.raw_path, self.BITEXT_FILE),
            encoding="utf-8", on_bad_lines="skip"
        )
        print(f"    Raw rows: {len(df):,}")
        df.columns = df.columns.str.strip()
        df = df.dropna(subset=["instruction", "response"])
        df = df.drop_duplicates(subset=["instruction"])
        print(f"    After dedup: {len(df):,}")
        df["instruction"] = df["instruction"].apply(DataLoader._clean_text)
        df["response"]    = df["response"].apply(DataLoader._clean_text)
        mask = (
            df["instruction"].apply(DataLoader._is_meaningful) &
            df["response"].apply(DataLoader._is_meaningful)
        )
        df = df[mask]
        print(f"    After quality filter: {len(df):,}")
        docs = []
        for _, row in df.iterrows():
            text = (
                f"Category: {row.get('category', 'General')}\n"
                f"Intent: {row.get('intent', 'N/A')}\n"
                f"Customer Question: {row.get('instruction', '')}\n"
                f"Answer: {row.get('response', '')}"
            )
            docs.append({
                "text": text,
                "source": "faq",
                "metadata": {
                    "category": str(row.get("category", "general")),
                    "intent":   str(row.get("intent", "N/A"))
                }
            })
        print(f"    [OK] Produced {len(docs):,} FAQ documents")
        return docs

    def load_ecommerce_intents(self):
        print("[+] Loading Dataset 3: Ecommerce FAQ Intents (JSON)...")
        with open(
            os.path.join(self.raw_path, self.INTENTS_FILE),
            "r", encoding="utf-8"
        ) as f:
            data = json.load(f)
        intents = data.get('intents', data) if isinstance(data, dict) else data
        print(f"    Total intent groups: {len(intents):,}")
        docs, skipped = [], 0
        for intent in intents:
            tag  = DataLoader._clean_text(str(intent.get('tag', 'general')))
            pats = [DataLoader._clean_text(p) for p in intent.get('patterns',  []) if p]
            ress = [DataLoader._clean_text(r) for r in intent.get('responses', []) if r]
            pats = [p for p in pats if DataLoader._is_meaningful(p, 2)]
            ress = [r for r in ress if DataLoader._is_meaningful(r, 3)]
            if not pats or not ress:
                skipped += 1
                continue
            text = (
                f"Topic: {tag}\n"
                f"Customer might ask: {'|'.join(pats)}\n"
                f"Best Answer: {ress[0]}"
            )
            if len(ress) > 1:
                text += f"\nAdditional Info: {' '.join(ress[1:])}"
            docs.append({
                "text": text,
                "source": "ecommerce_intents",
                "metadata": {
                    "tag":            tag,
                    "pattern_count":  str(len(pats)),
                    "response_count": str(len(ress))
                }
            })
        print(f"    Skipped: {skipped}")
        print(f"    [OK] Produced {len(docs):,} intent documents")
        return docs

    def load_all(self):
        print("=" * 50)
        print("  SMARTSERVE AI -- DATA LOADER STARTING")
        print("=" * 50)
        all_docs = []
        all_docs.extend(self.load_support_tickets())
        all_docs.extend(self.load_faq_data())
        all_docs.extend(self.load_ecommerce_intents())
        print("[SWEEP] Final safety sweep...")
        cleaned = 0
        for doc in all_docs:
            orig = doc['text']
            doc["text"] = re.sub(r"[{][{][^}]*[}][}]", "", doc["text"])
            doc["text"] = re.sub(r"[{][{][^}]*",       "", doc["text"])
            doc["text"] = re.sub(r"[{][^}]*[}]",       "", doc["text"])
            doc["text"] = re.sub(r"[{][^}]*$",         "", doc["text"])
            doc["text"] = re.sub(r"\s+",              " ", doc["text"]).strip()
            if doc['text'] != orig:
                cleaned += 1
        print(f"    Cleaned {cleaned} additional docs in sweep")
        print("=" * 50)
        print(f"  TOTAL DOCUMENTS : {len(all_docs):,}")
        for src, cnt in Counter(d['source'] for d in all_docs).items():
            print(f"    {src:<25} {cnt:>6,} docs")
        print("=" * 50)
        return all_docs

    def save_processed(self, documents):
        os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
        rows = [{"text": d["text"], "source": d["source"], **d["metadata"]}
                for d in documents]
        df = pd.DataFrame(rows)
        out = os.path.join(PROCESSED_DATA_PATH, 'knowledge_base.csv')
        df.to_csv(out, index=False, encoding="utf-8")
        print(f"[SAVED] {len(df):,} documents -> {out}")
        return out
