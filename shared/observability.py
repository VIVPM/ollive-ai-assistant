import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent.parent / "data" / "observability.db"

# Cost per 1K tokens (USD) — update as pricing changes
COST_PER_1K = {
    "Qwen/Qwen2.5-0.5B-Instruct": {"in": 0.0, "out": 0.0},                       # OSS — runs locally on your hardware, no per-token cost
    "google/gemma-2-9b-it": {"in": 0.00008, "out": 0.00008},                     # frontier — HF router, ~$0.08 / 1M tokens
    "google/gemma-2-9b-it:featherless-ai": {"in": 0.00008, "out": 0.00008},      # frontier — HF router, ~$0.08 / 1M tokens
    "meta-llama/Llama-3.3-70B-Instruct": {"in": 0.0001, "out": 0.00032},         # judge — HF router, ~$0.10 in / $0.32 out per 1M tokens
}


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(
        """CREATE TABLE IF NOT EXISTS calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            assistant TEXT NOT NULL,
            model TEXT NOT NULL,
            ts REAL NOT NULL,
            latency_ms REAL NOT NULL,
            in_tokens INTEGER NOT NULL,
            out_tokens INTEGER NOT NULL,
            est_cost_usd REAL NOT NULL
        )"""
    )
    conn.commit()
    return conn


class ObservabilityLogger:
    def log_call(
        self,
        assistant: str,
        model: str,
        latency_ms: float,
        in_tokens: int,
        out_tokens: int,
    ) -> None:
        rates = COST_PER_1K.get(model, {"in": 0.0, "out": 0.0})
        cost = (in_tokens / 1000) * rates["in"] + (out_tokens / 1000) * rates["out"]
        conn = _get_conn()
        conn.execute(
            "INSERT INTO calls (assistant, model, ts, latency_ms, in_tokens, out_tokens, est_cost_usd) VALUES (?,?,?,?,?,?,?)",
            (assistant, model, time.time(), latency_ms, in_tokens, out_tokens, cost),
        )
        conn.commit()
        conn.close()

    def get_summary(self, assistant: Optional[str] = None) -> dict:
        conn = _get_conn()
        where = "WHERE assistant=?" if assistant else ""
        params = (assistant,) if assistant else ()
        row = conn.execute(
            f"SELECT COUNT(*), AVG(latency_ms), SUM(est_cost_usd), SUM(in_tokens+out_tokens) FROM calls {where}",
            params,
        ).fetchone()
        conn.close()
        return {
            "calls": row[0] or 0,
            "avg_latency_ms": round(row[1] or 0, 1),
            "total_cost_usd": round(row[2] or 0, 6),
            "total_tokens": row[3] or 0,
        }


@contextmanager
def timed():
    """Context manager that yields a dict; sets 'ms' key on exit."""
    result = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["ms"] = (time.perf_counter() - start) * 1000
