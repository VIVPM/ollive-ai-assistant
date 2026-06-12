"""
HF Spaces — unified personal assistant. v3
OSS (Qwen2.5-0.5B) runs inside this Space; Frontier (Gemma 2 9B) calls the
Hugging Face Inference router. Set HF_TOKEN as a Space secret to enable Frontier.
"""
import os
import time
from threading import Thread

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OSS_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
FRONTIER_MODEL = "google/gemma-2-9b-it:featherless-ai"
HF_BASE_URL = "https://router.huggingface.co/v1"
MAX_NEW_TOKENS = 512

OSS_SYSTEM = (
    "You are a helpful, harmless, and honest personal assistant powered by "
    "Qwen2.5-0.5B running locally. You support multi-turn conversation. "
    "Answer clearly and concisely."
)
FRONTIER_SYSTEM = (
    "You are a helpful, harmless, and honest personal assistant powered by Gemma 2 9B. "
    "You support multi-turn conversation. Answer clearly and concisely."
)

MODELS = {
    "Frontier — Gemma 2 9B (HF Inference API)": "frontier",
    "OSS — Qwen2.5-0.5B (This Space)": "oss",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _text(content) -> str:
    """Safely extract text from Gradio content (str or multimodal list)."""
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content) if content else ""


def _history_to_messages(history: list, system: str) -> list:
    messages = [{"role": "system", "content": system}]
    for entry in history:
        if isinstance(entry, dict):
            messages.append({"role": entry["role"], "content": _text(entry["content"])})
    return messages


def _fold_system_into_user(messages: list) -> list:
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


# ---------------------------------------------------------------------------
# OSS model — loaded once at startup
# ---------------------------------------------------------------------------
print(f"Loading {OSS_MODEL_ID}...")
_tokenizer = AutoTokenizer.from_pretrained(OSS_MODEL_ID)
_model = AutoModelForCausalLM.from_pretrained(
    OSS_MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)
print("OSS model ready.")


def stream_oss(message: str, history: list):
    messages = _history_to_messages(history, OSS_SYSTEM)
    messages.append({"role": "user", "content": message})

    prompt: str = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)

    streamer = TextIteratorStreamer(
        _tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    Thread(
        target=_model.generate,
        kwargs=dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=_tokenizer.eos_token_id,
        ),
        daemon=True,
    ).start()

    for token in streamer:
        yield token


# ---------------------------------------------------------------------------
# Frontier model — Gemma 2 9B via HF Inference router
# ---------------------------------------------------------------------------
_hf_client = None


def _get_hf():
    global _hf_client
    if _hf_client is None:
        from openai import OpenAI
        api_key = os.getenv("HF_TOKEN")
        if not api_key:
            raise ValueError("HF_TOKEN not set. Add it as a Space secret.")
        _hf_client = OpenAI(api_key=api_key, base_url=HF_BASE_URL)
    return _hf_client


def stream_frontier(message: str, history: list):
    messages = _history_to_messages(history, FRONTIER_SYSTEM)
    messages.append({"role": "user", "content": message})

    stream = _get_hf().chat.completions.create(
        model=FRONTIER_MODEL,
        messages=_fold_system_into_user(messages),
        max_tokens=1024,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices:
            token = chunk.choices[0].delta.content or ""
            if token:
                yield token


# ---------------------------------------------------------------------------
# Observability (in-memory, resets on Space restart)
# ---------------------------------------------------------------------------
COST_PER_1K = {
    "frontier": {"in": 0.00008, "out": 0.00008},  # Gemma 2 9B via HF router — ~$0.08 / 1M tokens
    "oss":      {"in": 0.0,     "out": 0.0},       # local — runs on Space hardware, no per-token cost
}

_stats: dict = {
    "frontier": {"calls": 0, "total_ms": 0.0, "in_tokens": 0, "out_tokens": 0, "cost": 0.0},
    "oss":      {"calls": 0, "total_ms": 0.0, "in_tokens": 0, "out_tokens": 0, "cost": 0.0},
}


def _log(key: str, ms: float, in_tok: int, out_tok: int) -> None:
    rates = COST_PER_1K[key]
    cost = (in_tok / 1000) * rates["in"] + (out_tok / 1000) * rates["out"]
    s = _stats[key]
    s["calls"] += 1
    s["total_ms"] += ms
    s["in_tokens"] += in_tok
    s["out_tokens"] += out_tok
    s["cost"] += cost


def _stats_text(label: str) -> str:
    key = MODELS[label]
    s = _stats[key]
    avg_ms = round(s["total_ms"] / s["calls"], 1) if s["calls"] else 0
    return (
        f"Calls: {s['calls']}\n"
        f"Avg latency: {avg_ms} ms\n"
        f"Input tokens: {s['in_tokens']}\n"
        f"Output tokens: {s['out_tokens']}\n"
        f"Total tokens: {s['in_tokens'] + s['out_tokens']}\n"
        f"Est. cost: ${s['cost']:.6f}"
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="AI Personal Assistant") as demo:
    gr.Markdown(
        "## AI Personal Assistant\n"
        "Switch between **Frontier (Gemma 2 9B · HF Inference API)** and "
        "**OSS (Qwen2.5-0.5B · this Space)** using the dropdown."
    )

    with gr.Row():
        model_dd = gr.Dropdown(
            choices=list(MODELS.keys()),
            value=list(MODELS.keys())[0],
            label="Model",
            scale=3,
        )
        status_box = gr.Textbox(value="Ready.", label="Status", interactive=False, scale=2)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=480, label="Chat")
            with gr.Row():
                msg_box = gr.Textbox(placeholder="Ask me anything...", show_label=False, scale=4)
                send_btn = gr.Button("Send", variant="primary", scale=1)
            clear_btn = gr.Button("Clear conversation")

        with gr.Column(scale=1):
            gr.Markdown("### Session Stats")
            stats_box = gr.Textbox(label="Observability", lines=5, interactive=False)
            refresh_btn = gr.Button("Refresh stats")

    # ------------------------------------------------------------------
    def on_switch(label, history):
        return gr.update(), f"Switched to {label.split('—')[1].strip()}."

    def _submit(message, history, label):
        if not message.strip():
            yield history, "", gr.update(), gr.update()
            return

        key = MODELS[label]
        history = history + [{"role": "user", "content": message}]
        history = history + [{"role": "assistant", "content": ""}]
        yield history, "", "Generating...", gr.update()

        # Estimate input tokens from conversation (1 token ≈ 4 chars)
        all_text = " ".join(_text(e["content"]) for e in history[:-2]) + " " + message
        in_tok = max(1, len(all_text) // 4)

        start = time.perf_counter()
        out_tok = 0

        stream_fn = stream_frontier if key == "frontier" else stream_oss
        try:
            for token in stream_fn(message, history[:-2]):
                history[-1]["content"] += token
                out_tok += len(token.split())
                yield history, "", gr.update(), gr.update()
        except Exception as e:
            history[-1]["content"] = f"[Error] {e}"
            yield history, "", f"Error: {e}", gr.update()
            return

        ms = (time.perf_counter() - start) * 1000
        _log(key, ms, in_tok, out_tok)
        yield history, "", "Done.", _stats_text(label)

    model_dd.change(on_switch, [model_dd, chatbot], [chatbot, status_box])
    send_btn.click(_submit, [msg_box, chatbot, model_dd], [chatbot, msg_box, status_box, stats_box])
    msg_box.submit(_submit, [msg_box, chatbot, model_dd], [chatbot, msg_box, status_box, stats_box])
    clear_btn.click(lambda: ([], "", "Cleared."), outputs=[chatbot, msg_box, status_box])
    refresh_btn.click(_stats_text, inputs=[model_dd], outputs=[stats_box])

demo.launch()
