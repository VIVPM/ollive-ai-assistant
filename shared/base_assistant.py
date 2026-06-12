from abc import ABC, abstractmethod
from typing import List, Tuple


class BaseAssistant(ABC):
    """Abstract base for both OSS and frontier assistants."""

    @abstractmethod
    def chat(self, message: str, history: List[Tuple[str, str]]) -> str:
        """Return assistant reply given a user message and Gradio-format history."""

    def reset(self) -> None:
        """Override to clear any session state beyond history."""
