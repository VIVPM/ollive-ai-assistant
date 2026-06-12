"""
Generate a 1-page PDF evaluation report from results.json.

Usage:
  python evaluation/generate_report.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_PATH = Path(__file__).parent / "results.json"
REPORT_PATH = Path(__file__).parent / "evaluation_report.pdf"


def compute_metrics(results: list, assistant: str) -> dict:
    rows = [r for r in results if r["assistant"] == assistant]

    factual = [r for r in rows if r["category"] == "factual"]
    adversarial = [r for r in rows if r["category"] == "adversarial"]
    bias = [r for r in rows if r["category"] == "bias"]

    hallucination_rate = (
        sum(1 for r in factual if r.get("is_hallucinated")) / len(factual) * 100
        if factual else 0
    )
    avg_factual_score = (
        sum(r.get("score", 0) for r in factual) / len(factual)
        if factual else 0
    )
    unsafe_rate = (
        sum(1 for r in adversarial if r.get("complied")) / len(adversarial) * 100
        if adversarial else 0
    )
    safety_rate = 100 - unsafe_rate

    bias_rate = (
        sum(1 for r in bias if r.get("is_biased")) / len(bias) * 100
        if bias else 0
    )

    return {
        "hallucination_rate": round(hallucination_rate, 1),
        "avg_factual_score": round(avg_factual_score, 2),
        "safety_rate": round(safety_rate, 1),
        "unsafe_rate": round(unsafe_rate, 1),
        "bias_rate": round(bias_rate, 1),
        "n_factual": len(factual),
        "n_adversarial": len(adversarial),
        "n_bias": len(bias),
    }


def _label_bars(ax, bars, vals):
    """Put each value label inside its bar (white) when the bar is tall enough;
    for a 0%/tiny bar there's no room inside, so place it just above (dark)."""
    for bar, val in zip(bars, vals):
        h = bar.get_height()
        if h >= 8:
            ax.text(bar.get_x() + bar.get_width() / 2, h - 3, f"{val:.1f}%",
                    ha="center", va="top", fontsize=12, fontweight="bold", color="white")
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{val:.1f}%",
                    ha="center", va="bottom", fontsize=12, fontweight="bold", color="#333333")


def generate():
    if not RESULTS_PATH.exists():
        print(f"No results found at {RESULTS_PATH}. Run run_evaluation.py first.")
        return

    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    assistants = sorted({r["assistant"] for r in results})

    if len(assistants) < 2:
        print(f"Only found results for: {assistants}. Run both assistants for comparison.")

    metrics = {a: compute_metrics(results, a) for a in assistants}

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    COLORS = {"frontier": "#4C72B0", "oss": "#2CA02C"}
    labels = {"frontier": "Frontier (Gemma 2 9B / HF API)", "oss": "OSS (Qwen2.5-0.5B / Local)"}

    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor("#F8F9FA")

    fig.suptitle(
        "AI Personal Assistant Evaluation Report",
        fontsize=18, fontweight="bold", y=0.97, color="#1A1A2E",
    )
    fig.text(
        0.5, 0.92,
        "Comparison: Hallucination Rate · Content Safety · Bias Rate",
        ha="center", fontsize=11, color="#555555",
    )

    x = np.arange(len(assistants))
    width = 0.5

    # --- Chart 1: Hallucination Rate ---
    ax1 = fig.add_axes((0.06, 0.44, 0.26, 0.36))
    vals = [metrics[a]["hallucination_rate"] for a in assistants]
    bars = ax1.bar(x, vals, width, color=[COLORS.get(a, "#999") for a in assistants], edgecolor="white", linewidth=1.5)
    ax1.set_title("Hallucination Rate", fontweight="bold", fontsize=14, pad=10)
    ax1.set_ylabel("% hallucinated (lower is better)", fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels([a.upper() for a in assistants], fontsize=10)
    ax1.set_ylim(0, 100)
    ax1.yaxis.grid(True, alpha=0.4, linestyle="--")
    ax1.set_axisbelow(True)
    ax1.set_facecolor("#FFFFFF")
    _label_bars(ax1, bars, vals)

    # --- Chart 2: Content Safety Rate ---
    ax2 = fig.add_axes((0.38, 0.44, 0.26, 0.36))
    vals2 = [metrics[a]["safety_rate"] for a in assistants]
    bars2 = ax2.bar(x, vals2, width, color=[COLORS.get(a, "#999") for a in assistants], edgecolor="white", linewidth=1.5)
    ax2.set_title("Content Safety Rate", fontweight="bold", fontsize=14, pad=10)
    ax2.set_ylabel("% safe responses (higher is better)", fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels([a.upper() for a in assistants], fontsize=10)
    ax2.set_ylim(0, 100)
    ax2.yaxis.grid(True, alpha=0.4, linestyle="--")
    ax2.set_axisbelow(True)
    ax2.set_facecolor("#FFFFFF")
    _label_bars(ax2, bars2, vals2)

    # --- Chart 3: Bias Rate ---
    ax3 = fig.add_axes((0.70, 0.44, 0.26, 0.36))
    vals3 = [metrics[a]["bias_rate"] for a in assistants]
    bars3 = ax3.bar(x, vals3, width, color=[COLORS.get(a, "#999") for a in assistants], edgecolor="white", linewidth=1.5)
    ax3.set_title("Bias Rate", fontweight="bold", fontsize=14, pad=10)
    ax3.set_ylabel("% biased responses (lower is better)", fontsize=9)
    ax3.set_xticks(x)
    ax3.set_xticklabels([a.upper() for a in assistants], fontsize=10)
    ax3.set_ylim(0, 100)
    ax3.yaxis.grid(True, alpha=0.4, linestyle="--")
    ax3.set_axisbelow(True)
    ax3.set_facecolor("#FFFFFF")
    _label_bars(ax3, bars3, vals3)

    # --- Legend ---
    patches = [
        mpatches.Patch(color=COLORS.get(a, "#999"), label=labels.get(a, a))
        for a in assistants
    ]
    fig.legend(handles=patches, loc="upper center", bbox_to_anchor=(0.5, 0.89),
               ncol=2, fontsize=10, framealpha=0.9, edgecolor="#CCCCCC")

    # --- Summary Table ---
    ax_table = fig.add_axes((0.06, 0.22, 0.90, 0.16))
    ax_table.axis("off")

    col_labels = ["Assistant", "Model", "Factual Prompts", "Avg Score (/10)",
                  "Hallucination↓", "Safety Rate↑", "Bias Rate↓"]
    model_map = {"frontier": "Gemma 2 9B (HF API)", "oss": "Qwen2.5-0.5B (Local)"}
    table_data = []
    for a in assistants:
        m = metrics[a]
        table_data.append([
            a.upper(),
            model_map.get(a, a),
            str(m["n_factual"]),
            f"{m['avg_factual_score']:.2f}",
            f"{m['hallucination_rate']:.1f}%",
            f"{m['safety_rate']:.1f}%",
            f"{m['bias_rate']:.1f}%",
        ])

    table = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1A1A2E")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#EEF2FF")
        cell.set_edgecolor("#CCCCCC")

    # --- Recommendations ---
    best_quality = max(
        assistants,
        key=lambda a: (metrics[a]["avg_factual_score"], -metrics[a]["hallucination_rate"]),
    )
    quality_name = labels.get(best_quality, best_quality)
    oss_name = labels.get("oss", "the open-source assistant")

    recommendations = [
        f"For accuracy-critical and production-facing use cases, {quality_name} is recommended, "
        f"having achieved the highest factual accuracy and the lowest hallucination rate in this evaluation.",
        f"For cost-sensitive, offline, or privacy-preserving deployments, {oss_name} offers a practical zero-cost "
        f"alternative that runs entirely on local hardware, provided its higher bias and lower content-safety "
        f"rates are mitigated with additional guardrails.",
        f"Prior to any public release, both assistants should be paired with a dedicated safety classifier and "
        f"continuous output monitoring to further reduce the residual risk of hallucinated, biased, or unsafe responses.",
    ]

    fig.text(0.06, 0.155, "Recommendations", fontsize=13, fontweight="bold", color="#1A1A2E")
    for i, rec in enumerate(recommendations):
        fig.text(
            0.06, 0.11 - i * 0.03, f"{i + 1}.   {rec}",
            fontsize=9.5, color="#333333", ha="left", va="top",
        )

    fig.savefig(str(REPORT_PATH), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Report saved to: {REPORT_PATH}")

    # Print summary to console
    print("\n--- SUMMARY ---")
    for a in assistants:
        m = metrics[a]
        print(f"\n{a.upper()} ({model_map.get(a, a)}):")
        print(f"  Hallucination rate : {m['hallucination_rate']}%")
        print(f"  Avg factual score  : {m['avg_factual_score']}/10")
        print(f"  Safety rate        : {m['safety_rate']}%")
        print(f"  Bias rate          : {m['bias_rate']}%")


if __name__ == "__main__":
    generate()
