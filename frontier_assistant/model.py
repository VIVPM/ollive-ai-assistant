"""Frontier assistant — Gemma 2 9B via the Hugging Face Inference router (hosted API)."""
import os
import sys
import time
from pathlib import Path
from typing import Generator, List, Tuple

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.base_assistant import BaseAssistant
from shared.memory import ConversationMemory
from shared.observability import ObservabilityLogger, timed

load_dotenv()

MODEL = "google/gemma-2-9b-it:featherless-ai"
HF_BASE_URL = "https://router.huggingface.co/v1"
SYSTEM_PROMPT = (
    "You are a helpful, harmless, and honest personal assistant powered by Gemma 2 9B. "
    "You support multi-turn conversation and remember the context of this session. "
    "Answer clearly and concisely."
)

_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _fold_system_into_user(messages: List[dict]) -> List[dict]:
    """Gemma has no system role — fold the system prompt into the first user turn."""
    if not messages or messages[0]["role"] != "system":
        return messages
    system = messages[0]["content"]
    folded, merged = [], False
    for m in messages[1:]:
        if not merged and m["role"] == "user":
            folded.append({"role": "user", "content": f"{system}\n\n{m['content']}"})
            merged = True
        else:
            folded.append(m)
    return folded


class FrontierAssistant(BaseAssistant):
    def __init__(self):
        api_key = os.getenv("HF_TOKEN")
        if not api_key:
            raise ValueError("HF_TOKEN not set. Add it to your .env file.")
        self._client = OpenAI(api_key=api_key, base_url=HF_BASE_URL)
        self._memory = ConversationMemory(max_turns=10)
        self._logger = ObservabilityLogger()

    def chat(self, message: str, history: List[Tuple[str, str]], use_tools: bool = False) -> str:
        messages = self._memory.to_openai_messages(SYSTEM_PROMPT)
        messages.append({"role": "user", "content": message})

        with timed() as t:
            response = self._client.chat.completions.create(
                model=MODEL,
                messages=_fold_system_into_user(messages),
                max_tokens=1024,
            )

        reply = response.choices[0].message.content or ""
        usage = response.usage
        in_tok = usage.prompt_tokens if usage else sum(_count_tokens(str(m.get("content", ""))) for m in messages)
        out_tok = usage.completion_tokens if usage else _count_tokens(reply)

        self._logger.log_call("frontier", MODEL, t["ms"], in_tok, out_tok)
        self._memory.add(message, reply)
        return reply

    def stream_chat(self, message: str, history: List[Tuple[str, str]]) -> Generator[str, None, None]:
        messages = self._memory.to_openai_messages(SYSTEM_PROMPT)
        messages.append({"role": "user", "content": message})

        in_tok = sum(_count_tokens(str(m.get("content", ""))) for m in messages)

        start = time.perf_counter()
        stream = self._client.chat.completions.create(
            model=MODEL,
            messages=_fold_system_into_user(messages),
            max_tokens=1024,
            stream=True,
        )

        full_reply = ""
        api_in_tok = api_out_tok = 0
        for chunk in stream:
            if chunk.choices:
                token = chunk.choices[0].delta.content or ""
                if token:
                    full_reply += token
                    yield token
            if chunk.usage:
                api_in_tok = chunk.usage.prompt_tokens
                api_out_tok = chunk.usage.completion_tokens

        if api_in_tok:
            in_tok = api_in_tok
        out_tok = api_out_tok if api_out_tok else _count_tokens(full_reply)

        latency_ms = (time.perf_counter() - start) * 1000
        self._logger.log_call("frontier", MODEL, latency_ms, in_tok, out_tok)
        self._memory.add(message, full_reply)

    def reset(self) -> None:
        self._memory.clear()

    def get_stats(self) -> dict:
        return self._logger.get_summary("frontier")
