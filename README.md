# Ollive AI Personal Assistant

Two AI personal assistants compared on hallucination, bias, and content safety: an
**open-source model (Qwen2.5-0.5B) running locally** against a **hosted frontier model
(Gemma 2 9B via the Hugging Face Inference API)**. Responses are scored with an
LLM-as-judge (Llama 3.3 70B). The OSS model is also deployed publicly on HF Spaces.

**Live demo (HF Spaces):** [huggingface.co/spaces/VPM100/ollive-oss-assistant](https://huggingface.co/spaces/VPM100/ollive-oss-assistant)

## Models

| | OSS Assistant | Frontier Assistant |
|---|---|---|
| Model | Qwen2.5-0.5B-Instruct | Gemma 2 9B |
| Provider | **Local** (transformers, CPU/GPU) | Hugging Face Inference API (router) |
| Parameters | 0.5B | 9B |
| Purpose | Open-source assistant + public deployment | Hosted comparison baseline |

**Evaluation judge:** Llama 3.3 70B Instruct, via the Hugging Face Inference router.

## Quickstart

### 1. Clone and set up environment

```bash
git clone https://github.com/VIVPM/ollive-ai-assistant.git
cd ollive-ai-assistant
python -m venv .venv
.venv\Scripts\activate        # Windows  (use: source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
```

> The OSS model runs locally via `transformers`. `transformers` is pinned `<5.0`
> because 5.x requires `torch>=2.7` (it references `torch.float8_e8m0fnu`).

### 2. Configure your token

```bash
cp .env.example .env
```

Edit `.env` — only a Hugging Face token is required:
```
HF_TOKEN=your_hf_token        # get from huggingface.co/settings/tokens
```

The token must have access to `meta-llama/Llama-3.3-70B-Instruct` (the judge). The
Gemma frontier model is served through the `featherless-ai` provider on the HF router.

### 3. Run the unified app

```bash
python app.py
```

Opens at `http://127.0.0.1:7860`. Select **OSS — Qwen2.5-0.5B (Local)** or
**Frontier — Gemma 2 9B** from the dropdown. The first OSS message downloads and
loads the model (~1 GB), so it takes a moment; subsequent messages are fast.

### Run evaluation

```bash
python evaluation/run_evaluation.py --assistant frontier   # Gemma 2 9B (API)
python evaluation/run_evaluation.py --assistant oss        # Qwen2.5-0.5B (local CPU)
python evaluation/run_evaluation.py --assistant both       # runs both
python evaluation/generate_report.py                       # generates evaluation_report.pdf
```

## Project Structure

```
ollive-ai-assistant/
├── app.py                          # Unified app with model dropdown
├── oss_assistant/
│   ├── model.py                    # Qwen2.5-0.5B local inference (transformers) + streaming
│   └── guardrails.py               # Input/output safety filters
├── frontier_assistant/
│   ├── model.py                    # Gemma 2 9B via HF Inference router + streaming
│   └── guardrails.py               # Input/output safety filters
├── shared/
│   ├── base_assistant.py           # Abstract base class
│   ├── memory.py                   # Sliding window conversation memory
│   ├── observability.py            # SQLite latency/token/cost logger
│   └── tools.py                    # Calculator, datetime, web search
├── evaluation/
│   ├── prompts/                    # 100 evaluation prompts (JSON)
│   ├── evaluator.py                # LLM-as-judge (Llama 3.3 70B via HF router)
│   ├── run_evaluation.py           # Async evaluation runner
│   └── generate_report.py          # PDF report generator
└── deployment/
    └── hf_spaces/                  # Self-contained HF Spaces deployment (Qwen 0.5B local)
```

## Features

- **Multi-turn conversations** with sliding-window memory (last 10 turns)
- **Streaming** — tokens appear word-by-word in real time (local streamer for OSS, SSE for frontier)
- **Guardrails** — input blocklist + output safety check on both models
- **Observability** — per-call latency, token count, and cost logged to SQLite; displayed live in the UI
- **Model switcher** — swap between OSS and frontier mid-conversation; history carries over

## Architecture Decisions

**Why Qwen2.5-0.5B for the open-source assistant?**

It runs **locally** with no API dependency, which is the whole point of the OSS side —
the assistant you evaluate is the same one you can deploy. At 0.5B params (~1 GB) it
loads inside the free Hugging Face Spaces CPU tier (~512 MB–2 GB RAM, no GPU), so the
*evaluated* model and the *deployed* model are identical. It is instruction-tuned and
supports a native system role, multi-turn chat, and a clean chat template.

**Why Gemma 2 9B as the frontier model?**

The frontier side needs a more capable, *hosted* model to compare against. Gemma 2 9B
is Google DeepMind's mid-size instruction model, served through Hugging Face's Inference
router (OpenAI-compatible), so it needs no extra client code and reuses the same HF
token. The 0.5B-vs-9B comparison (~18×) tells a meaningful quality-vs-cost story between
a tiny local model and a larger hosted one. Gemma has no `system` role, so the system
prompt is folded into the first user turn.

**Why Llama 3.3 70B as the judge?**

A judge should be clearly stronger than both models under test. Llama 3.3 70B is large,
strong at structured/JSON output, and available on the HF Inference router with the same
token — so the entire stack runs on a single HF credential, no second provider.

**Why the HF Inference *router* (`router.huggingface.co`)?**

The legacy `api-inference.huggingface.co` serverless endpoint has been retired (its
hostname no longer resolves). The current OpenAI-compatible gateway is
`https://router.huggingface.co/v1`, which routes to providers (e.g. `featherless-ai`)
behind one API.

**How conversation memory works**

Each assistant keeps its own `ConversationMemory` — a deque of turn pairs capped at 10.
When the window fills, the oldest turn is dropped, keeping prompt length predictable.
When you switch models in the UI, the Gradio history is synced into the new model's
memory so the conversation continues without losing context.

## Tradeoffs

| Concern | Decision | Tradeoff |
|---|---|---|
| OSS inference | Qwen 0.5B local (transformers) | Free, fully offline, deployable; CPU latency higher, ~1 GB first-load |
| Frontier model | Gemma 2 9B via HF router | Free, no extra client; cold start on first call |
| Memory | In-context sliding window | Simple, no external DB; forgets turns beyond window |
| Guardrails | Regex blocklist + output check | No heavy classifier; may miss subtle harmful outputs |
| Evaluation | LLM-as-judge (Llama 3.3 70B) | Faster and cheaper than human eval; judge bias possible |
| OSS eval concurrency | 1 (single shared CPU model) | Deterministic, avoids contention; slower than the API side |

## Evaluation

Evaluated on 110 prompts: 50 factual, 30 adversarial, 30 bias probes.
Judge: Llama 3.3 70B Instruct (HF Inference router).

| Metric | OSS (Qwen2.5-0.5B) | Frontier (Gemma 2 9B) |
|---|---|---|
| Hallucination rate ↓ | 14.0% | **0.0%** |
| Avg factual score ↑ | 8.70 / 10 | **9.82 / 10** |
| Content safety rate ↑ | 66.7% | **96.7%** |
| Bias rate ↓ | 33.3% | **0.0%** |

The frontier model (Gemma 2 9B) leads on every axis, as expected from the ~18×
parameter gap — most notably on bias (0.0% vs 33.3%) and content safety (96.7% vs
66.7%). The 0.5B OSS model stays competitive on factual accuracy (8.70 vs 9.82) but
hallucinates and produces unsafe/biased responses far more often, which is the core
trade-off of running a tiny local model.

Reproduce with `python evaluation/run_evaluation.py --assistant both` then
`python evaluation/generate_report.py`. Full infographic report:
`evaluation/evaluation_report.pdf`.

## Cost and Latency

Logged to `shared/observability.db` during the evaluation run and shown live in the UI.

| | OSS (Qwen2.5-0.5B / local) | Frontier (Gemma 2 9B / HF API) |
|---|---|---|
| Cost per 1K tokens | $0.00 (runs on your hardware) | ~$0.00008 ($0.08 per 1M tokens, in & out) |
| Hosting | Local GPU/CPU | HF Inference router |
| Avg response latency | a few seconds on a GTX 1650 GPU (~20–31 tok/s greedy; ~5–6 tok/s on CPU) | ~3.3 s |
| Cold start | ~1 GB model load on first use | Provider cold start on first call |

The OSS model carries no API cost but is latency-bound by local hardware; the frontier
model is free on the HF tier and faster, at the cost of an external dependency.

## What I Would Improve with More Time

1. **Better guardrails** — integrate a dedicated safety classifier (LlamaGuard) instead of regex patterns
2. **Persistent memory** — replace the in-context window with a vector store (ChromaDB) for memory that survives session resets
3. **Quantized OSS inference** — load Qwen in 4-bit / GGUF (llama.cpp) for faster CPU latency
4. **Tool use** — wire up the calculator, datetime, and search tools for both assistants
