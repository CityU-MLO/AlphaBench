import os
from factors.lib.alpha158 import load_factors_alpha158
from agent.generator_qlib_search import call_qlib_search
from api.factor_eval_client import (
    batch_evaluate_factors_via_api,
    evaluate_factor_via_api,
)

from searcher.CoT.CoT_searcher import CoTSearcher
from searcher.ToT.ToT_searcher import ToTSearcher

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Callable, Dict, Iterable, Any, Optional
from tqdm.auto import tqdm


def to_cot_kwargs(rec):
    return {
        "seed": {"name": rec["name"], "expression": rec["expression"]},
        "run_name": rec["name"],
    }


def to_tot_kwargs(rec):
    return {
        "seed": {"name": rec["name"], "expression": rec["expression"]},
        "run_name": rec["name"],
    }


def run_batch(
    records: Iterable[dict],
    worker: Any = None,
    *,
    method: Optional[str] = None,
    func: Optional[Callable[..., Any]] = None,
    record_to_kwargs: Callable[[dict], Dict[str, Any]],
    key_fn: Callable[[dict], str] = lambda r: r.get("name", "<unnamed>"),
    common_kwargs: Optional[Dict[str, Any]] = None,
    num_workers: int = 4,
    show_progress: bool = True,
    executor: str = "thread",  # "thread" or "process"
):
    """
    Generic concurrent batch runner.

    Args:
        records: iterable of record dicts.
        worker: object whose method will be called (if `method` is provided).
        method: name of the method on `worker` to call per record. (e.g., "search_single_factor")
        func: alternative to `method`—a plain callable(**kwargs) per record.
        record_to_kwargs: maps each record -> kwargs dict for the call.
        key_fn: maps each record -> key used in outputs (default uses record["name"]).
        common_kwargs: kwargs applied to every call (merged with per-record kwargs; per-record wins on conflicts).
        num_workers: parallel workers.
        show_progress: show tqdm progress bar.
        executor: "thread" (I/O-bound) or "process" (CPU-bound).

    Returns:
        (results, errors): dicts keyed by `key_fn(record)`.
    """
    if (method is None) == (func is None):
        raise ValueError("Provide exactly one of `method` or `func`.")

    records = list(records)
    results, errors = {}, {}
    common_kwargs = common_kwargs or {}

    # Resolve callable
    if method:
        if worker is None:
            raise ValueError("`worker` must be provided when using `method`.")
        call = getattr(worker, method)
    else:
        call = func

    Exec = ThreadPoolExecutor if executor == "thread" else ProcessPoolExecutor

    with Exec(max_workers=num_workers) as ex:
        futures = {}
        for rec in records:
            per_kwargs = record_to_kwargs(rec)
            kwargs = {**common_kwargs, **per_kwargs}
            futures[ex.submit(call, **kwargs)] = rec

        pbar = tqdm(
            total=len(futures),
            disable=not show_progress,
            desc="Batch run",
            dynamic_ncols=True,
        )
        for fut in as_completed(futures):
            rec = futures[fut]
            k = key_fn(rec)
            try:
                results[k] = fut.result()
            except Exception as e:
                errors[k] = str(e)
            finally:
                pbar.update(1)
        pbar.close()

    return results, errors


def benchmark_main(
    model="deepseek-chat",
    temperature=1.75,
    enable_reason=True,
    local=False,
    local_port=8000,
    save_dir="./runs/T3_searching",
):

    """Main function to run the benchmark for searching factors."""
    os.makedirs(save_dir, exist_ok=True)
    standard_factors, compile_factors = load_factors_alpha158(
        exclude_var="vwap", collection=["kbar", "rolling"]
    )

    # Here you would typically set up your benchmark environment,
    # run the benchmarks, and collect results.
    # This is a placeholder for the actual benchmarking logic.
    print(
        "Load {} standard factors and {} compiled factors.".format(
            len(standard_factors), len(compile_factors)
        )
    )

    parsed_factor_pool = [
        {"name": factor.get("name"), "expr": factor.get("qlib_expression_default")}
        for factor in compile_factors.values()
    ]

    factor_performance = batch_evaluate_factors_via_api(parsed_factor_pool)

    filtered_performance = [
        {
            "name": result["name"],
            "expression": result["expression"],
            "metrics": {
                k: v
                for k, v in result["metrics"].items()
                if k in ["ir", "ic", "rank_ic", "rank_icir", "icir"]
            },
        }
        for result in factor_performance
    ]

    with open(f"{save_dir}/factor_seed_metrics.json", "w") as f:
        import json

        json.dump(filtered_performance, f, indent=4)

    # Start CoT searching
    cot_save_dir = os.path.join(save_dir, "CoT")
    os.makedirs(cot_save_dir, exist_ok=True)

    cot_searcher = CoTSearcher(
        evaluate_factor_fn=evaluate_factor_via_api,
        search_fn=call_qlib_search,
        model=model,
        temperature=temperature,
        enable_reason=enable_reason,
        local=local,
        local_port=local_port,
    )

    common_cot = {"rounds": 10, "verbose": False, "save_dir": cot_save_dir}

    results, errors = run_batch(
        records=filtered_performance,
        worker=cot_searcher,
        method="search_single_factor",
        record_to_kwargs=to_cot_kwargs,
        common_kwargs=common_cot,
        num_workers=4,
        show_progress=True,
        executor="thread",  # API/IO-bound typical for LLM calls
    )

    # Start ToT searching
    tot_save_dir = os.path.join(save_dir, "ToT")
    os.makedirs(tot_save_dir, exist_ok=True)

    common_tot = {"rounds": 10, "verbose": False, "save_dir": tot_save_dir, "N": 10}

    tot_searcher = ToTSearcher(
        evaluate_factor_fn=evaluate_factor_via_api,
        batch_evaluate_factors_fn=batch_evaluate_factors_via_api,
        search_fn=call_qlib_search,
        model=model,
        temperature=temperature,
        enable_reason=enable_reason,
        local=local,
        local_port=local_port,
    )

    results_tot, errors_tot = run_batch(
        records=filtered_performance,
        worker=tot_searcher,
        method="search_single_factor",  # whatever your method is called
        record_to_kwargs=to_tot_kwargs,
        common_kwargs=common_tot,
        num_workers=4,
        show_progress=True,
        executor="thread",  # API/IO-bound typical for LLM calls
    )

    import pdb

    pdb.set_trace()  # Debugging point to inspect the loaded factors


if __name__ == "__main__":
    benchmark_main()
