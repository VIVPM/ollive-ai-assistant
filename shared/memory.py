from collections import deque
from typing import List, Tuple


class ConversationMemory:
    """
    Sliding-window conversation memory.
    Keeps the last `max_turns` user/assistant pairs.
    Converts to OpenAI-format messages for API calls.
    """

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self._history: deque = deque(maxlen=max_turns)

    def add(self, user_msg: str, assistant_msg: str) -> None:
        self._history.append((user_msg, assistant_msg))

    def to_openai_messages(self, system_prompt: str) -> List[dict]:
        messages = [{"role": "system", "content": system_prompt}]
        for user_msg, assistant_msg in self._history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        return messages

    def to_gradio_history(self) -> List[Tuple[str, str]]:
        return list(self._history)

    def clear(self) -> None:
        self._history.clear()

    def __len__(self) -> int:
        return len(self._history)
