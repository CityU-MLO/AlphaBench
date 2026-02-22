"""Quick smoke test for the FFO server."""
import json
import sys
import requests

BASE = "http://127.0.0.1:19777"

def pprint(label, status, data):
    print("=" * 70)
    print(f"[{label}]  HTTP {status}")
    print(json.dumps(data, indent=2, ensure_ascii=False)[:2000])
    if isinstance(data, list):
        ok = sum(1 for x in data if x.get("success"))
        print(f"  => ok={ok}  fail={len(data)-ok}")
    print()

def test_health():
    r = requests.get(f"{BASE}/health", timeout=10)
    pprint("HEALTH", r.status_code, r.json())
    assert r.status_code == 200

def test_check():
    r = requests.post(f"{BASE}/factors/check",
                      json={"expression": "Mean($close, 20)"},
                      timeout=30)
    pprint("CHECK", r.status_code, r.json())

def test_fast_single():
    payload = {
        "expression": "$close",
        "start": "2022-01-01",
        "end": "2022-03-01",
        "market": "csi300",
        "fast": True,
        "use_cache": True,
        "topk": 50,
        "n_drop": 5,
        "timeout": 120,
    }
    r = requests.post(f"{BASE}/factors/eval", json=payload, timeout=180)
    pprint("FAST-SINGLE", r.status_code, r.json())

def test_fast_batch():
    payload = {
        "expression": [
            "$close",
            "Mean($close, 20)",
            "Std($close, 20)",
        ],
        "start": "2022-01-01",
        "end": "2022-03-01",
        "market": "csi300",
        "fast": True,
        "use_cache": True,
        "topk": 50,
        "n_drop": 5,
        "timeout": 120,
    }
    r = requests.post(f"{BASE}/factors/eval", json=payload, timeout=300)
    pprint("FAST-BATCH(3)", r.status_code, r.json())

def test_fast_named():
    payload = {
        "expression": {
            "alpha_close": "$close",
            "alpha_mean20": "Mean($close, 20)",
        },
        "start": "2022-01-01",
        "end": "2022-03-01",
        "market": "csi300",
        "fast": True,
        "use_cache": True,
    }
    r = requests.post(f"{BASE}/factors/eval", json=payload, timeout=300)
    pprint("FAST-NAMED", r.status_code, r.json())

def test_full_single():
    payload = {
        "expression": "$close",
        "start": "2022-01-01",
        "end": "2022-03-01",
        "market": "csi300",
        "fast": False,
        "use_cache": True,
        "topk": 50,
        "n_drop": 5,
        "timeout": 120,
        "n_jobs_backtest": 4,
    }
    r = requests.post(f"{BASE}/factors/eval", json=payload, timeout=600)
    pprint("FULL-SINGLE", r.status_code, r.json())

if __name__ == "__main__":
    tests = [test_health, test_check, test_fast_single, test_fast_batch, test_fast_named]
    if "--full" in sys.argv:
        tests.append(test_full_single)

    failed = []
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"!!! {t.__name__} FAILED: {e}\n")
            failed.append(t.__name__)

    print("=" * 70)
    if failed:
        print(f"FAILED: {failed}")
    else:
        print("ALL PASSED")
