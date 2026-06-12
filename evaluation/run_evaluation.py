"""
Run all evaluation prompts against one or both assistants.

Usage:
  python evaluation/run_evaluation.py --assistant frontier
  python evaluation/run_evaluation.py --assistant oss
  python evaluation/run_evaluation.py --assistant both   (default)

Frontier (Gemma 2 9B) is API-bound so its prompts run concurrently. OSS
(Qwen2.5-0.5B) runs locally on CPU, so it is kept at concurrency 1 to avoid
contending for the single shared model.
"""
import argparse
import asyncio
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

PROMPTS_DIR = Path(__file__).parent / "prompts"
RESULTS_PATH = Path(__file__).parent / "results.json"

CONCURRENCY = {"frontier": 5, "oss": 1}

# Evaluate in batches, then pause — keeps the HF judge from being throttled by a
# burst of calls (a throttled judge errors out and silently scores everything 0).
BATCH_SIZE = 10
BATCH_SLEEP_S = 15


def load_prompts(name: str) -> list:
    return json.loads((PROMPTS_DIR / f"{name}.json").read_text(encoding="utf-8"))


def get_assistant(name: str):
    if name == "frontier":
        from frontier_assistant.model import FrontierAssistant
        return FrontierAssistant()
    if name == "oss":
        from oss_assistant.model import OSSAssistant
        return OSSAssistant()
    raise ValueError(f"Unknown assistant: {name}")


def _sync_query(assistant_name: str, prompt: str) -> str:
    """Fresh assistant instance per call — no memory cross-contamination between prompts."""
    assistant = get_assistant(assistant_name)
    try:
        return assistant.chat(prompt, [], use_tools=False)
    except TypeError:
        return assistant.chat(prompt, [])
    except Exception as e:
        return f"[ERROR] {e}"


async def _eval_one(
    sem: asyncio.Semaphore,
    executor: ThreadPoolExecutor,
    assistant_name: str,
    category: str,
    scorer,
    item: dict,
) -> dict:
    loop = asyncio.get_event_loop()

    async with sem:
        response = await loop.run_in_executor(executor, _sync_query, assistant_name, item["prompt"])

        score_args = (
            [item["prompt"], item["answer"], response]
            if category == "factual"
            else [item["prompt"], response]
        )
        scores = await loop.run_in_executor(executor, scorer, *score_args)

    result = {
        "assistant": assistant_name,
        "category": category,
        "sub_category": item.get("category", ""),
        "id": item["id"],
        "prompt": item["prompt"],
        "response": response,
        **scores,
    }
    if category == "factual":
        result["ground_truth"] = item["answer"]
        print(f"  {item['id']}... score={scores['score']} hallucinated={scores['is_hallucinated']}")
    elif category == "adversarial":
        print(f"  {item['id']}... complied={scores['complied']} severity={scores['severity']}")
    else:
        print(f"  {item['id']}... is_biased={scores['is_biased']} type={scores['bias_type']}")

    return result


async def _eval_in_batches(sem, executor, assistant_name, category, scorer, items) -> list:
    """Evaluate items in batches of BATCH_SIZE, sleeping BATCH_SLEEP_S between batches
    so the HF judge isn't hit by a burst of calls (which trips its rate limit)."""
    results = []
    for start in range(0, len(items), BATCH_SIZE):
        batch = items[start:start + BATCH_SIZE]
        results.extend(await asyncio.gather(*[
            _eval_one(sem, executor, assistant_name, category, scorer, item)
            for item in batch
        ]))
        done = start + len(batch)
        if done < len(items):
            print(f"  ...{done}/{len(items)} done — sleeping {BATCH_SLEEP_S}s to avoid judge throttling...")
            await asyncio.sleep(BATCH_SLEEP_S)
    return results


async def run_for_assistant_async(assistant_name: str) -> list:
    from evaluation.evaluator import score_adversarial, score_bias, score_factual

    print(f"\n{'='*60}")
    print(f"Evaluating: {assistant_name.upper()}")
    print("=" * 60)

    concurrency = CONCURRENCY[assistant_name]
    sem = asyncio.Semaphore(concurrency)

    with ThreadPoolExecutor(max_workers=concurrency * 2) as executor:
        print("\n[1/3] Factual prompts...")
        factual = await _eval_in_batches(sem, executor, assistant_name, "factual", score_factual, load_prompts("factual"))

        print("\n[2/3] Adversarial prompts...")
        adversarial = await _eval_in_batches(sem, executor, assistant_name, "adversarial", score_adversarial, load_prompts("adversarial"))

        print("\n[3/3] Bias prompts...")
        bias = await _eval_in_batches(sem, executor, assistant_name, "bias", score_bias, load_prompts("bias"))

    return list(factual) + list(adversarial) + list(bias)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--assistant",
        choices=["frontier", "oss", "both"],
        default="both",
        help="Which assistant(s) to evaluate",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-evaluate even if an assistant already has a complete set of results",
    )
    args = parser.parse_args()

    if RESULTS_PATH.exists():
        results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        print(f"Loaded {len(results)} existing results.")
    else:
        results = []

    # An assistant counts as "done" only when every prompt across all categories is scored.
    total_prompts = sum(len(load_prompts(c)) for c in ("factual", "adversarial", "bias"))

    targets = ["frontier", "oss"] if args.assistant == "both" else [args.assistant]

    for name in targets:
        already_done = sum(1 for r in results if r["assistant"] == name)
        if already_done >= total_prompts and not args.force:
            print(f"\n{name} already fully evaluated ({already_done}/{total_prompts}) — skipping (use --force to redo).")
            continue
        # Re-running: drop this assistant's old/partial rows first so results aren't duplicated or left stale.
        results = [r for r in results if r["assistant"] != name]
        results.extend(asyncio.run(run_for_assistant_async(name)))

    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {len(results)} total results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
