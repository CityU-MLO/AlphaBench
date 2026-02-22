# test_eval.py
import json
import requests

BASE_URL = "http://127.0.0.1:19320"
EVAL_URL = f"{BASE_URL}/factors/eval"

EXPRESSIONS = [
    "$close",
    "Mean($close, 20)",
    "Std($close, 20)",
    "$close/Mean($close, 20)",
    "($close-Mean($close, 20))/Mean($close, 20)",
    "$volume/Mean($volume, 20)",
    "($high-$low)/$close",
    "Corr($close, Log($volume+1), 20)",
    "Delta($close, 20)",
    "Rank($close, 20)",
]


def post_eval(payload: dict, timeout: int = 180):
    resp = requests.post(EVAL_URL, json=payload, timeout=timeout)
    try:
        data = resp.json()
    except Exception:
        data = {"_raw": resp.text}
    return resp.status_code, data


def pretty_print(title: str, status: int, data):
    print("=" * 80)
    print(title)
    print("HTTP", status)
    print(json.dumps(data, indent=2, ensure_ascii=False))

    if isinstance(data, list):
        ok = sum(1 for x in data if x.get("success") is True)
        fail = len(data) - ok
        print(f"\nSummary: ok={ok}, fail={fail}")
    print()


def run_case(title: str, common: dict, payload_extra: dict, timeout: int):
    payload = {**common, **payload_extra}
    status, data = post_eval(payload, timeout=timeout)
    pretty_print(title, status, data)


def main():
    # Base config (adjust to your qlib env)
    common = {
        "start": "2022-01-01",
        "end": "2022-03-01",
        "market": "csi300",
        "label": "close_return",
        "use_cache": True,
        "topk": 50,
        "n_drop": 5,
        "timeout": 120,  # eval timeout per factor
        "n_jobs_backtest": 4,  # threads for portfolio backtest when fast=False
    }

    # ===============================
    # FAST MODE: only IC, no backtest
    # ===============================
    common_fast = {**common, "fast": True}

    run_case(
        "FAST-1: single expression (string)",
        common_fast,
        {"expression": EXPRESSIONS[0]},
        timeout=300,
    )

    run_case(
        "FAST-2: multiple expressions (list, no names)",
        common_fast,
        {"expression": EXPRESSIONS},
        timeout=600,
    )

    named = {f"F{i+1}": e for i, e in enumerate(EXPRESSIONS)}
    run_case(
        "FAST-3: multiple expressions (dict name->expr)",
        common_fast,
        {"expression": named},
        timeout=600,
    )

    mixed = [
        EXPRESSIONS[0],
        {"alpha_mean20": EXPRESSIONS[1]},
        EXPRESSIONS[2],
        {"alpha_rank20": EXPRESSIONS[-1]},
    ]
    run_case(
        "FAST-4: mixed list (strings + dicts)",
        common_fast,
        {"expression": mixed},
        timeout=600,
    )

    # ==========================================
    # FULL MODE: IC + portfolio backtest (threads)
    # ==========================================
    common_full = {**common, "fast": False}

    run_case(
        "FULL-1: single expression (string) + portfolio backtest",
        common_full,
        {"expression": EXPRESSIONS[0]},
        timeout=900,
    )

    run_case(
        "FULL-2: multiple expressions (list, no names) + portfolio backtest",
        common_full,
        {"expression": EXPRESSIONS},
        timeout=1800,
    )

    run_case(
        "FULL-3: multiple expressions (dict name->expr) + portfolio backtest",
        common_full,
        {"expression": named},
        timeout=1800,
    )

    run_case(
        "FULL-4: mixed list (strings + dicts) + portfolio backtest",
        common_full,
        {"expression": mixed},
        timeout=1800,
    )


if __name__ == "__main__":
    main()
