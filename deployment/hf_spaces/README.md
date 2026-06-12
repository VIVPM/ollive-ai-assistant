---
title: Ollive Oss Assistant
emoji: 🌍
colorFrom: red
colorTo: purple
sdk: gradio
sdk_version: 6.15.2
app_file: app.py
pinned: false
---

# Ollive OSS Assistant — Live Demo

A Gradio chat app that switches between two assistants via a dropdown:

- **OSS — Qwen2.5-0.5B** — runs **locally inside this Space** via `transformers` (free, no API).
- **Frontier — Gemma 2 9B** — served through the **Hugging Face Inference router** (`router.huggingface.co`).

Features: multi-turn chat, streaming responses, and live per-session observability
(latency, token counts, estimated cost).

## Configuration

Set one secret in **Settings → Variables and secrets**:

| Secret | Purpose |
|--------|---------|
| `HF_TOKEN` | Authenticates the Frontier (Gemma 2 9B) calls to the HF Inference router. |

The OSS model needs no key — it is downloaded and run inside the Space.

> Part of the [Ollive AI Personal Assistant](https://github.com/VIVPM/ollive-ai-assistant)
> project. See the main repository for the evaluation pipeline and architecture notes.
