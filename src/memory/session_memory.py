import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from collections import defaultdict, deque


class SimpleMemory:
    """
    Lightweight conversation memory using a deque (sliding window).
    Stores last K turns per session — no LangChain dependency needed.
    Each turn = one HumanMessage + one AIMessage pair.
    """
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        # Each session stores a deque of (human, ai) tuples
        self._turns: deque = deque(maxlen=window_size)

    def add_turn(self, human: str, ai: str):
        self._turns.append((human, ai))

    def get_history_text(self) -> str:
        if not self._turns:
            return ""
        lines = []
        for human, ai in self._turns:
            lines.append(f"Customer: {human}")
            lines.append(f"Assistant: {ai}")
        return "\n".join(lines)

    def turn_count(self) -> int:
        return len(self._turns)

    def clear(self):
        self._turns.clear()


class SessionMemoryManager:
    """
    Manages per-session conversation memory.
    Each session_id gets its own isolated SimpleMemory instance.
    In-memory only — resets on server restart.
    Production upgrade: swap defaultdict for Redis.
    """

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self._sessions: dict = {}

    def _get_or_create(self, session_id: str) -> SimpleMemory:
        if session_id not in self._sessions:
            self._sessions[session_id] = SimpleMemory(self.window_size)
        return self._sessions[session_id]

    def save_turn(self, session_id: str, human_message: str, ai_message: str):
        """Saves one conversation turn to session memory."""
        mem = self._get_or_create(session_id)
        mem.add_turn(human_message, ai_message)

    def get_history_as_text(self, session_id: str) -> str:
        """Returns conversation history as formatted text for prompt injection."""
        if session_id not in self._sessions:
            return ""
        return self._sessions[session_id].get_history_text()

    def turn_count(self, session_id: str) -> int:
        """Returns number of completed turns in a session."""
        if session_id not in self._sessions:
            return 0
        return self._sessions[session_id].turn_count()

    def clear_session(self, session_id: str):
        """Clears memory for a specific session."""
        if session_id in self._sessions:
            del self._sessions[session_id]

    def session_count(self) -> int:
        """Returns total number of active sessions."""
        return len(self._sessions)
