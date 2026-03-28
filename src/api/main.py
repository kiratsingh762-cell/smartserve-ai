import sys
import os
import re
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.rag.pipeline import RAGPipeline
from src.as400.connector import AS400Connector


# ─────────────────────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
#
# CONCEPT - Pydantic Models:
# Pydantic validates the shape of data automatically.
# If the UI sends {"message": 123} (number instead of string),
# FastAPI rejects it before it even reaches your code.
# This is called "data validation at the boundary."
# ─────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    answer: str
    sources: list
    confidence: str
    escalate_to_human: bool
    as400_data: dict = {}


# ─────────────────────────────────────────────────────────────
# APP INITIALIZATION
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="SmartServe AI API",
    description="Intelligent Customer Support — RAG + AS400",
    version="1.0.0"
)

# CONCEPT - CORS (Cross-Origin Resource Sharing):
# By default browsers block requests between different ports.
# Our UI runs on port 8501, API on port 8000 — different origins.
# CORS middleware tells the API: "accept requests from anywhere."
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize components ONCE at startup — not per request
# CONCEPT - Why once? Loading the vector store and LLM takes ~3 seconds.
# If we reloaded them for every customer message, each reply would
# take 3+ seconds just for initialization. Load once, reuse always.
print("[API] Loading RAG pipeline...")
rag    = RAGPipeline()
as400  = AS400Connector()
print("[API] SmartServe AI API ready!")


# ─────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.get("/")
def health_check():
    """Health check — call this to verify the server is running."""
    return {
        "status": "SmartServe AI is running",
        "version": "1.0.0",
        "vectors": "28,480",
        "model": "gpt-4o"
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Main chat endpoint — called for every customer message.

    FLOW:
    1. Detect if message contains an Order ID → query AS400
    2. Inject live AS400 data into the question context
    3. Run RAG pipeline → get AI answer
    4. Decide if escalation to human agent is needed
    5. Return structured response
    """
    try:
        message = request.message.strip()
        as400_data = {}

        if not message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        # ── STEP 1: AS400 Order Lookup ────────────────────────
        # CONCEPT - Regex for Order ID detection:
        # r'\bORD\d{5}\b' matches: word boundary + "ORD" + exactly 5 digits
        # \b = word boundary (so "WORD00001" doesn't match)
        # This detects patterns like "ORD00042" or "ord00042" in any message
        order_match = re.search(r'\b(ORD\d{5})\b', message.upper())

        if order_match:
            order_id   = order_match.group(1)
            order_info = as400.query_order_status(order_id)

            if order_info.get("found"):
                as400_data = order_info
                # Inject real-time data into the question so the AI
                # can reference it in its answer
                message = (
                    f"{message}\n\n"
                    f"[LIVE AS400 DATA] Order {order_id}: "
                    f"Status={order_info['status']}, "
                    f"Product={order_info['product']}, "
                    f"Tracking={order_info['tracking']}, "
                    f"Order Date={order_info['order_date']}"
                )

        # ── STEP 2: RAG Answer ────────────────────────────────
        result = rag.get_answer(message)

        # ── STEP 3: Escalation Logic ──────────────────────────
        # CONCEPT - When to escalate to a human:
        # We escalate when:
        # a) AI confidence is low (not enough context retrieved)
        # b) Customer explicitly asks for a human
        # c) Sensitive keywords detected (legal, complaint, refund demand)
        escalate_keywords = [
            "human", "agent", "representative", "person",
            "operator", "manager", "supervisor", "complaint",
            "legal", "lawsuit", "refund my money"
        ]
        should_escalate = (
            result["confidence"] == "low" or
            any(kw in message.lower() for kw in escalate_keywords)
        )

        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            confidence=result["confidence"],
            escalate_to_human=should_escalate,
            as400_data=as400_data
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/order/{order_id}")
def get_order(order_id: str):
    """Direct AS400 order lookup by Order ID."""
    result = as400.query_order_status(order_id.upper())
    if not result.get("found"):
        raise HTTPException(status_code=404, detail=f"Order {order_id} not found")
    return result


@app.get("/health")
def detailed_health():
    """Detailed health check for monitoring."""
    try:
        count = rag.vectorstore._collection.count()
        return {
            "api":          "online",
            "rag_pipeline": "online",
            "vector_store": "online",
            "vectors":      count,
            "as400":        "simulated/online"
        }
    except Exception as e:
        return {"api": "online", "error": str(e)}