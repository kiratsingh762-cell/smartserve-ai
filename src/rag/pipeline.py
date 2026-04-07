import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from src.knowledge_base.vectorstore import VectorStoreManager
from src.memory.session_memory import SessionMemoryManager
from config.settings import OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE, TOP_K_RESULTS


SYSTEM_PROMPT = """You are SmartServe AI, the official virtual assistant for Cole-Parmer.
You represent Cole-Parmer directly. Always refer to Cole-Parmer in first person:
- Say "our website", "our products", "our team", "we offer"
- NEVER say "their website", "Cole-Parmer's website", or refer to the company as a third party
- You ARE Cole-Parmer support — speak as a member of the Cole-Parmer team

CONVERSATION RULES:
1. Answer the customer's question using the context provided below.
2. If context is partial, use general best-practice knowledge to give a helpful response.
3. Use conversation history to maintain context — reference prior messages when relevant.
4. Be professional, empathetic, and concise. Max 150 words per response.
5. If the customer seems frustrated, acknowledge their frustration first.
6. For order-specific queries, ask for the Order ID if not already provided.
7. Never reveal that you are using a knowledge base or retrieved documents.

ENDING THE CONVERSATION:
- Do NOT add "Is there anything else I can help you with?" to every single response.
- Only ask "Is there anything else I can help you with?" when:
  a) You have fully resolved the customer's issue, AND
  b) The conversation feels like it is wrapping up naturally
- If the customer says anything like "No thanks", "That's all", "That's it",
  "I'm good", "No, thank you", "All done", or similar closing phrases —
  respond ONLY with:
  "Thank you for contacting Cole-Parmer. Have a wonderful day! Goodbye!"
  Do not add anything else after that closing message.
- For follow-up questions or ongoing troubleshooting, just answer naturally
  without appending the closing question every time.

LANGUAGE:
- Use "our" not "their" for anything Cole-Parmer related
- Use "we" not "they" when referring to Cole-Parmer actions or offerings
- Examples:
  WRONG: "You can visit Cole-Parmer's website for more information"
  RIGHT: "You can visit our website at coleparmer.com for more information"

  WRONG: "Cole-Parmer offers a wide range of products"
  RIGHT: "We offer a wide range of products"

  WRONG: "Their support team will assist you"
  RIGHT: "Our support team will be happy to assist you"

CONVERSATION HISTORY (last 5 turns):
{chat_history}

CONTEXT FROM KNOWLEDGE BASE:
{context}

CUSTOMER MESSAGE:
{question}

YOUR RESPONSE:"""


class RAGPipeline:
    """Memory-aware RAG pipeline with improved persona and conversation logic."""

    def __init__(self):
        print("[RAG] Initializing RAG Pipeline with Memory...")
        self.vs_manager = VectorStoreManager()
        self.vectorstore = self.vs_manager.load_vectorstore()
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K_RESULTS}
        )
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            openai_api_key=OPENAI_API_KEY
        )
        self.memory_manager = SessionMemoryManager(window_size=5)
        print("[RAG] Pipeline ready!")

    def _format_docs(self, docs: list) -> str:
        formatted = []
        for i, doc in enumerate(docs, 1):
            source   = doc.metadata.get("source", "knowledge_base")
            category = doc.metadata.get("category", "")
            product  = doc.metadata.get("product", "")
            label_parts = [f"Source {i}", source]
            if product and product != "N/A":
                label_parts.append(f"Product: {product}")
            if category:
                label_parts.append(f"Category: {category}")
            formatted.append(f"[{' | '.join(label_parts)}]\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)

    def _is_closing_message(self, message: str) -> bool:
        """Detects if the customer is ending the conversation."""
        closing_phrases = [
            "no thanks", "no thank you", "that's all", "thats all",
            "that's it", "thats it", "i'm good", "im good",
            "all done", "nothing else", "no more", "goodbye",
            "bye", "that will be all", "that will be it",
            "i'm fine", "im fine", "no need", "not anymore",
            "that helped", "problem solved", "issue resolved"
        ]
        msg_lower = message.lower().strip()
        return any(phrase in msg_lower for phrase in closing_phrases)

    def get_answer(self, question: str, session_id: str = "default") -> dict:
        if not question or not question.strip():
            return {
                "answer": "I did not receive a question. Could you please rephrase?",
                "sources": [], "confidence": "low", "chunks": [], "turn_count": 0
            }

        # If customer is closing the conversation, return farewell immediately
        if self._is_closing_message(question):
            farewell = "Thank you for contacting Cole-Parmer. Have a wonderful day! Goodbye!"
            self.memory_manager.save_turn(session_id, question, farewell)
            return {
                "answer":     farewell,
                "sources":    [],
                "confidence": "high",
                "chunks":     [],
                "turn_count": self.memory_manager.turn_count(session_id)
            }

        source_docs = self.retriever.invoke(question)
        context     = self._format_docs(source_docs)
        chat_history = self.memory_manager.get_history_as_text(session_id)

        prompt_text = SYSTEM_PROMPT.format(
            chat_history=chat_history if chat_history else "No previous messages.",
            context=context,
            question=question
        )

        response = self.llm.invoke([HumanMessage(content=prompt_text)])
        answer   = response.content

        self.memory_manager.save_turn(session_id, question, answer)

        sources = list(set(
            doc.metadata.get("source", "N/A") for doc in source_docs
        ))

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
            "chunks":     [doc.page_content[:100] for doc in source_docs],
            "turn_count": self.memory_manager.turn_count(session_id)
        }

    def clear_session(self, session_id: str):
        self.memory_manager.clear_session(session_id)

    def test_pipeline(self):
        session = "test_fixes_session"
        print("\n" + "=" * 60)
        print("  TEST -- Persona Fix + Conversation Ending")
        print("=" * 60)

        questions = [
            ("Q1 -- Product recommendation (persona test)",
             "Can you recommend a thermometer from your catalog?"),
            ("Q2 -- Follow-up troubleshooting",
             "My laptop screen is flickering"),
            ("Q3 -- Another follow-up (no closing question expected)",
             "What if those steps don't work?"),
            ("Q4 -- Closing message (farewell expected)",
             "No thanks, that's all"),
        ]

        for label, question in questions:
            print(f"\n[{label}]")
            print(f"Customer: {question}")
            print("-" * 60)
            result = self.get_answer(question, session_id=session)
            print(f"Turn     : {result['turn_count']}")
            print(f"Answer   :\n{result['answer']}")
            print("=" * 60)
