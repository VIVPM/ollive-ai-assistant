"""OSS assistant — Qwen2.5-0.5B-Instruct running locally via transformers (no API)."""
import sys
import time
from pathlib import Path
from threading import Lock, Thread
from typing import Generator, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.base_assistant import BaseAssistant
from shared.memory import ConversationMemory
from shared.observability import ObservabilityLogger, timed

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM_PROMPT = (
    "You are a helpful, harmless, and honest personal assistant powered by "
    "Qwen2.5-0.5B running locally. You support multi-turn conversation and "
    "remember the context of this session. Answer clearly and concisely."
)
MAX_NEW_TOKENS = 512

# Weights are heavy to load — keep one shared copy across all instances so the
# evaluation runner (which spins up a fresh assistant per prompt) doesn't reload
# the model 100 times. Memory stays per-instance, so there's no cross-contamination.
_model = None
_tokenizer = None
_load_lock = Lock()


def _load():
    global _model, _tokenizer
    if _model is None:
        with _load_lock:
            if _model is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                # fp16 only pays off on Ampere+ (compute capability >= 8.0). On older
                # cards like the GTX 1650 (Turing, no fast-fp16 tensor cores) fp32 is
                # ~40% faster, so choose dtype by GPU capability.
                if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
                    dtype = torch.float16
                else:
                    dtype = torch.float32
                _tokenizer = AutoTokenizer.from_pretrained(MODEL)
                _model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype).to(device)
                print(f"[oss] Qwen2.5-0.5B loaded on {device} ({str(dtype).replace('torch.', '')}).")
    return _model, _tokenizer


class OSSAssistant(BaseAssistant):
    def __init__(self):
        self._memory = ConversationMemory(max_turns=10)
        self._logger = ObservabilityLogger()

    def _build_inputs(self, message: str):
        model, tokenizer = _load()
        messages = self._memory.to_openai_messages(SYSTEM_PROMPT)
        messages.append({"role": "user", "content": message})
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        return model, tokenizer, inputs

    def _gen_kwargs(self, tokenizer, sample: bool) -> dict:
        kw: dict = {"max_new_tokens": MAX_NEW_TOKENS, "pad_token_id": tokenizer.eos_token_id}
        if sample:
            # Interactive chat — a bit of variety.
            kw.update(do_sample=True, temperature=0.7, top_p=0.9)
        else:
            # Evaluation — greedy is deterministic (reproducible scores) and faster.
            # Reset the sampling params to their neutral defaults (Qwen bakes non-default
            # values into its generation_config); equal-to-default silences the
            # "ignored in greedy mode" warnings without passing None (which crashes generate).
            kw.update(do_sample=False, temperature=1.0, top_p=1.0, top_k=50)
        return kw

    def chat(self, message: str, history: List[Tuple[str, str]], use_tools: bool = False) -> str:
        model, tokenizer, inputs = self._build_inputs(message)

        with timed() as t:
            output = model.generate(**inputs, **self._gen_kwargs(tokenizer, sample=False))

        gen_ids = output[0][inputs["input_ids"].shape[-1]:]
        reply = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        in_tok = int(inputs["input_ids"].shape[-1])
        out_tok = int(gen_ids.shape[-1])
        self._logger.log_call("oss", MODEL, t["ms"], in_tok, out_tok)
        self._memory.add(message, reply)
        return reply

    def stream_chat(self, message: str, history: List[Tuple[str, str]]) -> Generator[str, None, None]:
        model, tokenizer, inputs = self._build_inputs(message)
        in_tok = int(inputs["input_ids"].shape[-1])

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        start = time.perf_counter()
        Thread(
            target=model.generate,
            kwargs=dict(**inputs, streamer=streamer, **self._gen_kwargs(tokenizer, sample=True)),
            daemon=True,
        ).start()

        full_reply = ""
        for token in streamer:
            full_reply += token
            yield token

        latency_ms = (time.perf_counter() - start) * 1000
        out_tok = len(tokenizer(full_reply)["input_ids"])
        self._logger.log_call("oss", MODEL, latency_ms, in_tok, out_tok)
        self._memory.add(message, full_reply)

    def reset(self) -> None:
        self._memory.clear()

    def get_stats(self) -> dict:
        return self._logger.get_summary("oss")
