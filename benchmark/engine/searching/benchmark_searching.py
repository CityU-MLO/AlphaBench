import argparse
import os
from pathlib import Path
from factors.lib.alpha158 import load_factors_alpha158, load_factors_alpha158_names
from agent.generator_qlib_search import call_qlib_search
from api.factor_eval_client import (
    batch_evaluate_factors_via_api,
    evaluate_factor_via_api,
)

from searcher.CoT.CoT_searcher import CoTSearcher
from searcher.ToT.ToT_searcher import ToTSearcher
from searcher.EA.EA_searcher import EA_Searcher

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Callable, Dict, Iterable, Any, Optional
from tqdm.auto import tqdm

import yaml


def load_config():
    parser = argparse.ArgumentParser(description="AlphaBench Search Config")

    # main config file
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to YAML configuration file"
    )

    # override options
    parser.add_argument(
        "--model_name", type=str, default=None, help="Override model name in config"
    )
    parser.add_argument(
        "--model_local",
        type=lambda x: str(x).lower() == "true",
        default=None,
        help="Override model local flag (True/False)",
    )
    parser.add_argument(
        "--local_port", type=int, default=None, help="Override local port in config"
    )
    parser.add_argument(
        "--market", type=str, default="csi300", help="Market for factor evaluation"
    )
    parser.add_argument("--save_dir", type=str, default="./runs/T3_deepseek")

    args = parser.parse_args()

    # load yaml config
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    # apply overrides
    if args.model_name is not None:
        config["model"]["name"] = args.model_name
    if args.model_local is not None:
        config["model"]["local"] = args.model_local
    if args.local_port is not None:
        config["model"]["local_port"] = args.local_port

    config["save_dir"] = args.save_dir
    return config


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
    num_workers: int = 16,
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
    config,
    # model="deepseek-chat",
    enable_reason=False,
    # local=False,
    # local_port=8000,
    # save_dir="./runs/T3_searching",
):

    model = config["model"]["name"]
    local = config["model"].get("local", False)
    local_port = config["model"].get("local_port", 8000)
    temperature = config["model"].get("temperature", 1.5)
    market = config.get("market", "csi300")

    num_workers = 4
    save_dir = config["save_dir"]

    """Main function to run the benchmark for searching factors."""
    os.makedirs(save_dir, exist_ok=True)
    standard_factors, compile_factors = load_factors_alpha158(
        exclude_var="vwap", collection=["kbar", "rolling", "price"]
    )

    # Here you would typically set up your benchmark environment,
    # run the benchmarks, and collect results.
    # This is a placeholder for the actual benchmarking logic.
    print(
        "Load {} standard factors and {} compiled factors.".format(
            len(standard_factors), len(compile_factors)
        )
    )

    factor_groups = load_factors_alpha158_names()
    kbar_names = [fac["name"] for fac in factor_groups["kbar"]]
    price_names = [fac["name"] for fac in factor_groups["price"]]
    rolling_names = [fac["name"] for fac in factor_groups["rolling"]]

    # import pdb;pdb.set_trace()
    parsed_factor_pool = [
        {
            "name": factor.get("name"),
            "expression": factor.get("qlib_expression_default"),
        }
        for factor in compile_factors.values()
    ]

    factor_performance = batch_evaluate_factors_via_api(
        parsed_factor_pool, market=market
    )

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

    if config.get("cot", {}).get("enable", True):
        print("CoT searching enabled.")
        # Start CoT searching
        cot_save_dir = os.path.join(save_dir, "CoT")
        os.makedirs(cot_save_dir, exist_ok=True)
        filtered_performance_cot = [
            rec
            for rec in filtered_performance
            if rec["name"] in kbar_names + price_names
        ]

        rounds = config.get("cot", {}).get("rounds", 10)
        cot_searcher = CoTSearcher(
            evaluate_factor_fn=evaluate_factor_via_api,
            search_fn=call_qlib_search,
            model=model,
            temperature=temperature,
            enable_reason=enable_reason,
            local=local,
            local_port=local_port,
        )

        common_cot = {"rounds": rounds, "verbose": True, "save_dir": cot_save_dir}

        results, errors = run_batch(
            records=filtered_performance_cot,
            worker=cot_searcher,
            method="search_single_factor",
            record_to_kwargs=to_cot_kwargs,
            common_kwargs=common_cot,
            num_workers=num_workers,
            show_progress=True,
            executor="thread",  # API/IO-bound typical for LLM calls
        )

    if config.get("tot", {}).get("enable", True):
        print("ToT searching enabled.")
        tot_names = [""]
        filtered_performance_tot = [
            rec
            for rec in filtered_performance
            if rec["name"] in kbar_names + price_names
        ]
        tot_save_dir = os.path.join(save_dir, "ToT")
        os.makedirs(tot_save_dir, exist_ok=True)

        N = config.get("tot", {}).get("size", 6)
        rounds = config.get("tot", {}).get("rounds", 3)
        common_tot = {
            "rounds": rounds,
            "verbose": False,
            "save_dir": tot_save_dir,
            "N": N,
        }

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
        # import pdb;pdb.set_trace()
        results_tot, errors_tot = run_batch(
            records=filtered_performance_tot,
            worker=tot_searcher,
            method="search_single_factor",  # whatever your method is called
            record_to_kwargs=to_tot_kwargs,
            common_kwargs=common_tot,
            num_workers=num_workers,
            show_progress=True,
            executor="thread",  # API/IO-bound typical for LLM calls
        )

    # Start EA searching
    if config.get("ea", {}).get("enable", True):
        print("EA searching enabled.")
        filtered_performance_ea = [
            rec for rec in filtered_performance if rec["name"] in rolling_names
        ]
        ea_save_dir = os.path.join(save_dir, "EA")
        os.makedirs(ea_save_dir, exist_ok=True)

        mutation_rate = config.get("ea", {}).get("mutation_rate", 0.3)
        crossover_rate = config.get("ea", {}).get("crossover_rate", 0.7)
        generations = config.get("ea", {}).get("generations", 10)
        generate_size = config.get("ea", {}).get("generate_size", 20)
        population_size = config.get("ea", {}).get("population_size", 20)

        # parsed_factor_pool = [
        #     {"name": f.get("name"), "expression": f.get("qlib_expression_default")}
        #     for f in compile_factors.values()
        # ]

        # # Evaluate baseline metrics (IC, RankIC, etc.) via your API
        # perf = batch_evaluate_factors_via_api(parsed_factor_pool)
        # for i, p in enumerate(perf):
        #     parsed_factor_pool[i]["metrics"] = p.get("metrics", {})

        # Build and run EA searcher
        searcher = EA_Searcher(
            batch_evaluate_factors_fn=batch_evaluate_factors_via_api,
            search_fn=call_qlib_search,
            model=model,
            temperature=temperature,
            enable_reason=enable_reason,
            local=local,
            local_port=local_port,
            save_dir=ea_save_dir,
            seeds_top_k=12,
        )

        # import pdb;pdb.set_trace()
        summary = searcher.search_population(
            filtered_performance_ea,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            N=generate_size,
            rounds=generations,
            pool_size=population_size,
            verbose=True,
            save_pickle=True,
        )

        print(f"EA search completed. Results saved to {ea_save_dir}")


if __name__ == "__main__":
    config = load_config()

    benchmark_main(config)
