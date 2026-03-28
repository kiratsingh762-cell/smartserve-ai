import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.knowledge_base.vectorstore import VectorStoreManager
from config.settings import OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE, TOP_K_RESULTS


# ─────────────────────────────────────────────────────────────────────
# THE SYSTEM PROMPT
#
# CONCEPT - Why the system prompt is critical:
# This is the PERSONALITY and RULES of your AI agent.
# It tells GPT-4o:
#   WHO it is       → SmartServe AI, your company's support agent
#   WHAT it can do  → Answer from provided context only
#   WHAT to avoid   → Never make up information
#   HOW to respond  → Professional, empathetic, structured
#   WHEN to escalate→ Low confidence = hand off to human
#
# The {context} and {question} are PLACEHOLDERS filled at runtime.
# This template pattern is called a "prompt template."
# ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are SmartServe AI, a professional customer support assistant.
You help customers with product issues, technical problems, orders, and general queries.

STRICT RULES:
1. Always prioritize the context provided below when answering.
2. If the context is partial or unclear, combine it with general
   best-practice troubleshooting knowledge to give a helpful response.
   Only say you cannot help if the topic is completely outside support scope.
3. Be professional, empathetic, and concise. Max 150 words per response.
4. If the customer seems frustrated, acknowledge their frustration first.
5. Always end your response with: "Is there anything else I can help you with?"
6. For order-specific queries, ask for the customer's Order ID if not provided.
7. Never reveal that you are using a knowledge base or retrieved documents.

CONTEXT FROM KNOWLEDGE BASE:
{context}

CUSTOMER QUESTION:
{question}

YOUR RESPONSE:"""


class RAGPipeline:
    """
    The complete RAG (Retrieval-Augmented Generation) pipeline.

    CONCEPT - LangChain LCEL (LangChain Expression Language):
    LangChain uses the | (pipe) operator to chain steps together.
    Output of each step flows as input to the next — like an assembly line.

    Our chain:
    Step 1: {"context": retriever, "question": passthrough}
            Retriever fetches top-4 relevant chunks from ChromaDB
            Passthrough sends the original question unchanged

    Step 2: prompt
            Fills {context} and {question} into SYSTEM_PROMPT template

    Step 3: llm
            Sends the filled prompt to GPT-4o, gets response back

    Step 4: StrOutputParser
            Extracts just the text string from GPT-4o's response object

    This entire chain runs in ONE line: self.chain.invoke(question)
    """

    def __init__(self):
        print("[RAG] Initializing RAG Pipeline...")

        # Load the vector store from disk
        self.vs_manager = VectorStoreManager()
        self.vectorstore = self.vs_manager.load_vectorstore()

        # CONCEPT - Retriever:
        # A retriever wraps the vector store and makes it work
        # inside LangChain chains. search_kwargs={"k": 4} means
        # "fetch 4 most relevant chunks per query"
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K_RESULTS}
        )

        # CONCEPT - temperature=0:
        # Temperature controls how "creative" the AI is.
        # 0   = deterministic, consistent, factual (ideal for support)
        # 0.7 = more varied, creative (good for writing/brainstorming)
        # 1.0 = very random (good for poetry, bad for support!)
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            openai_api_key=OPENAI_API_KEY
        )

        self.prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)

        # Build the LCEL chain
        self.chain = (
            {
                "context":  self.retriever | self._format_docs,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        print("[RAG] Pipeline ready!")

    def _format_docs(self, docs: list) -> str:
        """
        Formats retrieved chunks into a clean string for the prompt.

        CONCEPT - Why format matters:
        The LLM receives context as plain text inside the prompt.
        Clear formatting (numbered sections, source labels) helps
        the LLM understand WHERE each piece of information came from
        and give a more accurate, structured answer.
        """
        formatted = []
        for i, doc in enumerate(docs, 1):
            source   = doc.metadata.get('source', 'knowledge_base')
            category = doc.metadata.get('category', '')
            product  = doc.metadata.get('product', '')

            # Build a clean label for this chunk
            label_parts = [f"Source {i}", source]
            if product and product != 'N/A':
                label_parts.append(f"Product: {product}")
            if category:
                label_parts.append(f"Category: {category}")
            label = " | ".join(label_parts)

            formatted.append(f"[{label}]\n{doc.page_content}")

        return "\n\n---\n\n".join(formatted)

    def get_answer(self, question: str) -> dict:
        """
        Main method: takes a customer question, returns AI answer + metadata.

        Returns a dict with:
          answer     → the AI's response text
          sources    → which data sources were used
          confidence → "high" if 3+ chunks found, "low" if fewer
          chunks     → the actual retrieved chunks (for debugging/transparency)
        """
        if not question or not question.strip():
            return {
                "answer": "I didn't receive a question. Could you please rephrase?",
                "sources": [],
                "confidence": "low",
                "chunks": []
            }

        # Get the answer from the full chain
        answer = self.chain.invoke(question)

        # Also retrieve source docs separately for metadata
        source_docs = self.retriever.invoke(question)

        # Build unique sources list
        sources = list(set(
            doc.metadata.get('source', 'N/A')
            for doc in source_docs
        ))

        # Confidence scoring:
        # high   = 3+ relevant chunks found → AI has good context
        # medium = 2 chunks found → AI has some context
        # low    = 0-1 chunks → AI may not have enough to answer well
        if len(source_docs) >= 3:
            confidence = "high"
        elif len(source_docs) >= 2:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "answer":     answer,
            "sources":    sources,
            "confidence": confidence,
            "chunks":     [doc.page_content[:100] for doc in source_docs]
        }

    def test_pipeline(self):
        """
        Runs 3 test queries to verify the full pipeline works.
        Call this after initialization to confirm everything is working.
        """
        test_questions = [
            "My device is not turning on, what should I do?",
            "How do I cancel my order?",
            "How do I create an account on your website?"
        ]

        print("\n" + "=" * 60)
        print("  RAG PIPELINE TEST -- 3 Sample Queries")
        print("=" * 60)

        for i, question in enumerate(test_questions, 1):
            print(f"\n[TEST {i}] Q: {question}")
            print("-" * 60)
            result = self.get_answer(question)
            print(f"Confidence : {result['confidence']}")
            print(f"Sources    : {result['sources']}")
            print(f"Answer     :\n{result['answer']}")
            print("=" * 60)