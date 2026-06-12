"""
Unified personal assistant — select Frontier or OSS model from the dropdown.
"""
import sys
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

from frontier_assistant.guardrails import check_input as frontier_check_input
from frontier_assistant.model import FrontierAssistant
from oss_assistant.guardrails import check_input as oss_check_input
from oss_assistant.model import OSSAssistant

MODELS = {
    "Frontier — Gemma 2 9B (HF Inference API)": "frontier",
    "OSS — Qwen2.5-0.5B (Local)": "oss",
}

# Lazy-loaded singletons
_assistants: dict = {}


def get_assistant(label: str):
    key = MODELS[label]
    if key not in _assistants:
        if key == "frontier":
            _assistants[key] = FrontierAssistant()
        else:
            _assistants[key] = OSSAssistant()
    return _assistants[key], key


def get_stats_text(label: str) -> str:
    if MODELS[label] not in _assistants:
        return "No calls yet."
    assistant, _ = get_assistant(label)
    s = assistant.get_stats()
    return (
        f"Calls: {s['calls']}\n"
        f"Avg latency: {s['avg_latency_ms']} ms\n"
        f"Total tokens: {s['total_tokens']}\n"
        f"Est. cost: ${s['total_cost_usd']:.6f}"
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="AI Personal Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## AI Personal Assistant\nSwitch between models using the dropdown below.")

    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=list(MODELS.keys()),
            value=list(MODELS.keys())[0],
            label="Model",
            scale=3,
        )
        status_box = gr.Textbox(
            value="Ready.",
            label="Status",
            interactive=False,
            scale=2,
        )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=480, label="Chat")
            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Ask me anything...",
                    show_label=False,
                    scale=4,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            clear_btn = gr.Button("Clear conversation")

        with gr.Column(scale=1):
            gr.Markdown("### Session Stats")
            stats_box = gr.Textbox(
                label="Observability",
                lines=6,
                interactive=False,
            )
            refresh_btn = gr.Button("Refresh stats")

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _extract_text(content) -> str:
        """Gradio 6 content can be a string or a list of multimodal parts."""
        if isinstance(content, list):
            return "".join(p.get("text", "") for p in content if isinstance(p, dict))
        return str(content) if content else ""

    def _sync_history_to_memory(assistant, history: list) -> None:
        """Rebuild assistant memory from Gradio chat history."""
        assistant.reset()
        i = 0
        while i < len(history) - 1:
            u, a = history[i], history[i + 1]
            if u.get("role") == "user" and a.get("role") == "assistant":
                assistant._memory.add(
                    _extract_text(u["content"]),
                    _extract_text(a["content"]),
                )
                i += 2
            else:
                i += 1

    def on_model_change(label, current_history):
        """Switch model — keep conversation visible and carry context over."""
        key = MODELS[label]
        if key in _assistants:
            _sync_history_to_memory(_assistants[key], current_history)
        model_name = label.split("—")[1].strip() if "—" in label else label
        # Return gr.update() for chatbot so Gradio keeps its current value intact
        return gr.update(), f"Switched to {model_name}. History carried over."

    def on_clear(label):
        """Explicitly clear conversation for the active model."""
        if MODELS[label] in _assistants:
            _assistants[MODELS[label]].reset()
        return [], "", "Conversation cleared."

    def _submit(message, history, label):
        if not message.strip():
            yield history, "", gr.update(), gr.update()
            return

        key = MODELS[label]
        check_input = frontier_check_input if key == "frontier" else oss_check_input
        is_safe, reason = check_input(message)
        if not is_safe:
            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": f"[Blocked] {reason}"},
            ]
            yield history, "", f"Blocked: {reason}", gr.update()
            return

        # Load assistant (may trigger OSS model download on first use)
        fresh_load = key not in _assistants
        if fresh_load:
            model_name = label.split("—")[1].strip() if "—" in label else label
            yield history, "", f"Loading {model_name}...", gr.update()

        assistant, _ = get_assistant(label)

        # Sync existing UI history into memory if this is a fresh load or new switch
        if fresh_load and len(history) > 0:
            _sync_history_to_memory(assistant, history)

        history = history + [{"role": "user", "content": message}]
        history = history + [{"role": "assistant", "content": ""}]
        yield history, "", "Generating...", gr.update()

        for token in assistant.stream_chat(message, []):
            history[-1]["content"] += token
            yield history, "", gr.update(), gr.update()

        yield history, "", "Done.", get_stats_text(label)

    def refresh_stats(label):
        return get_stats_text(label)

    model_dropdown.change(
        on_model_change,
        inputs=[model_dropdown, chatbot],
        outputs=[chatbot, status_box],
    )

    send_btn.click(
        _submit,
        inputs=[msg_box, chatbot, model_dropdown],
        outputs=[chatbot, msg_box, status_box, stats_box],
    )
    msg_box.submit(
        _submit,
        inputs=[msg_box, chatbot, model_dropdown],
        outputs=[chatbot, msg_box, status_box, stats_box],
    )
    clear_btn.click(
        on_clear,
        inputs=[model_dropdown],
        outputs=[chatbot, msg_box, status_box],
    )
    refresh_btn.click(
        refresh_stats,
        inputs=[model_dropdown],
        outputs=[stats_box],
    )


if __name__ == "__main__":
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
