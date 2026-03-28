# 🤖 SmartServe AI — Intelligent Customer Support System

> A production-grade AI chatbot that answers customer queries using 
> RAG (Retrieval-Augmented Generation) powered by GPT-4o, with live 
> AS400/IBM i integration for real-time order lookups.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.2-green)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-teal)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)

---

## 🎯 What It Does

| Scenario | How It Works |
|---|---|
| Technical Support Query | RAG retrieves relevant past tickets + FAQ, GPT-4o generates grounded answer |
| Live Order Status | Detects Order ID in message, queries AS400 DB2, returns real-time data |
| Human Escalation | Detects low confidence or escalation keywords, routes to human agent |

---

## 🏗️ System Architecture
Customer Message
│
▼
[FastAPI Backend :8000]
│
├──► [RAG Pipeline]
│ │
│ ├──► Embed query → ChromaDB Vector Search
│ │ └──► Top 4 relevant chunks retrieved
│ │
│ └──► GPT-4o generates grounded answer
│
├──► [AS400 Connector]
│ └──► Live order/inventory lookup (DB2 simulation)
│
└──► [Escalation Engine]
└──► Routes to human agent when needed
│
▼
[Streamlit UI :8501]

---

## 📊 Knowledge Base Stats

| Dataset | Source | Documents | Purpose |
|---|---|---|---|
| Customer Support Tickets | Kaggle | 1,848 | Past resolved issues |
| Bitext FAQ (27K) | Kaggle | 12,437 | FAQ & help center |
| Ecommerce Intents | Kaggle | 97 | Product & order intents |
| **Total Vectors** | ChromaDB | **28,480** | Searchable AI knowledge |

---

## 🚀 Tech Stack

| Layer | Technology |
|---|---|
| AI Brain | OpenAI GPT-4o |
| Retrieval | LangChain + ChromaDB |
| Embeddings | OpenAI text-embedding-3-small |
| Backend API | FastAPI + Uvicorn |
| Chat UI | Streamlit |
| Legacy System | AS400/IBM i DB2 (simulated) |
| Data Source | Kaggle Datasets |
| Language | Python 3.11 |

---

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.11+
- OpenAI API key ([get one here](https://platform.openai.com))
- Kaggle account ([get API token here](https://www.kaggle.com/settings))

### Installation

# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/smartserve-ai.git
cd smartserve-ai

# 2. Create virtual environment
python -m venv smartserve_env
smartserve_env\Scripts\activate   # Windows
# source smartserve_env/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 5. Download Kaggle datasets
kaggle datasets download -d suraj520/customer-support-ticket-dataset -p data/raw --unzip
kaggle datasets download -d bitext/bitext-gen-ai-chatbot-customer-support-dataset -p data/raw --unzip
kaggle datasets download -d walterebhota/ecommerce-dataset-for-nlpchatbot -p data/raw --unzip

# 6. Build the AI knowledge base (one-time, ~10 mins)
python build_knowledge_base.py

# 7. Launch the system (2 terminals)
uvicorn src.api.main:app --reload --port 8000   # Terminal 1
streamlit run ui/chatbot.py                      # Terminal 2
```

### Access
- **Chat UI:** http://localhost:8501
- **API Docs:** http://localhost:8000/docs

---

## 📁 Project Structure
smartserve-ai/
├── src/
│ ├── ingestion/ # Data loading & cleaning
│ ├── knowledge_base/ # Chunking & vector store
│ ├── rag/ # LangChain RAG pipeline
│ ├── as400/ # AS400/DB2 integration
│ └── api/ # FastAPI REST backend
├── ui/ # Streamlit chat interface
├── data/
│ ├── raw/ # Kaggle datasets (not committed)
│ ├── processed/ # Cleaned knowledge base
│ └── vectordb/ # ChromaDB vectors (not committed)
├── config/ # Central configuration
├── notebooks/ # Data exploration
├── build_knowledge_base.py # One-time KB builder
└── requirements.txt

---

## 👨‍💼 Built By

**Kirat Singh** — AI/Data Science Portfolio Project  
📍 Mumbai, India | 🎯 Targeting / AI Engineer / AI Automation / Data Science Roles  
[LinkedIn](www.linkedin.com/in/kirat-singh-9230b5a5)