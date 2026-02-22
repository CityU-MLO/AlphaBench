# test_check.py
import json
import requests

BASE_URL = "http://127.0.0.1:19777"
CHECK_URL = f"{BASE_URL}/factors/check"

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


def post_check(payload: dict, timeout: int = 60):
    resp = requests.post(CHECK_URL, json=payload, timeout=timeout)
    try:
        data = resp.json()
    except Exception:
        data = {"_raw": resp.text}
    return resp.status_code, data


def main():
    print("== Test 1: single expression (string) ==")
    status, data = post_check({"expression": EXPRESSIONS[0], "timeout": 30})
    print("HTTP", status)
    print(json.dumps(data, indent=2, ensure_ascii=False))
    print()

    print("== Test 2: multiple expressions (list, no names) ==")
    status, data = post_check({"expression": EXPRESSIONS, "timeout": 30})
    print("HTTP", status)
    # Expected: list of results (even if server returns list for single too)
    print(json.dumps(data, indent=2, ensure_ascii=False))
    print()

    print("== Test 3: multiple expressions (dict name->expr) ==")
    named = {f"F{i+1}": e for i, e in enumerate(EXPRESSIONS)}
    status, data = post_check({"expression": named, "timeout": 30})
    print("HTTP", status)
    print(json.dumps(data, indent=2, ensure_ascii=False))
    print()

    # Optional: quick summary if response is a list
    if isinstance(data, list):
        ok = sum(1 for x in data if x.get("success") is True)
        fail = len(data) - ok
        print(f"Summary (last run): ok={ok}, fail={fail}")


if __name__ == "__main__":
    main()
