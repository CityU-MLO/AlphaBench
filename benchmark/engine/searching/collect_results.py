import pickle
import json
import os
import sys
from typing import Dict, Any


def analyze_tot_checkpoint(summary: Dict[str, Any], top_k: int = 3) -> None:
    """
    Analyze a ToT search checkpoint summary.
    Prints per-round statistics and global best.

    Args:
        summary: dict loaded from pickle (the checkpoint returned by ToTSearcher)
        top_k: how many top candidates to display per depth
    """
    seed = summary.get("seed", {})
    seed_metrics = summary.get("seed_metrics", {})
    print("=" * 80)
    print(f"Seed: {seed.get('name')} => {seed.get('expression')}")
    print("Seed metrics:", seed_metrics)
    print("=" * 80)

    history = summary.get("history", [])
    for rec in history:
        depth = rec.get("depth")
        print(f"\n[Depth {depth}]")
        parent = rec.get("parent", {})
        print(f" Parent: {parent.get('name')} => {parent.get('expression')}")
        print(f" Candidates generated: {len(rec.get('candidates', []))}")
        print(f" Survivors kept: {len(rec.get('survivors', []))}")
        if rec.get("ranking"):
            print(" Top candidates:")
            for i, (nm, mx) in enumerate(rec["ranking"][:top_k], 1):
                ic = mx.get("ic")
                ric = mx.get("rank_ic")
                ir = mx.get("ir")
                print(f"   {i}. {nm} | IC={ic:.4f}, RankIC={ric:.4f}, IR={ir:.4f}")
        if rec.get("best_of_node"):
            nm, mx = rec["best_of_node"]
            print(f" Best-of-node: {nm} | {mx}")

    print("\n" + "=" * 80)
    best = summary.get("best", {})
    print("Global Best Factor:")
    print(f"  {best.get('name')} => {best.get('expression')}")
    print("  Metrics:", best.get("metrics", {}))
    print(f"Elapsed time: {summary.get('elapsed_sec'):.2f} sec")
    print("=" * 80)
    
if __name__ == "__main__":
    with open("./runs/tot_search/tot_parallel_search_Seed_KLEN.pkl", "rb") as f:
        ckpt = pickle.load(f)
    analyze_tot_checkpoint(ckpt)