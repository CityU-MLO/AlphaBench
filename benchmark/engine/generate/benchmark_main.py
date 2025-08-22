import csv
from datetime import datetime
from itertools import count
import logging
import math
import os
import json
import pickle
import shutil
import sys

import numpy as np
import argparse

from pathlib import Path

import pandas as pd
from agent.generator_qlib import call_gen_qlib_factors, batch_call_gen_qlib_factors
from agent.compiler import apply_parameters_to_template

from agent.qlib_contrib.qlib_expr_parsing import FactorParser

from benchmark.engine.utils import (
    get_factor_performance_batch,
    similarity_factor_output,
)
from benchmark.engine.generate.eval_fitness import start_eval_fitness
from benchmark.engine.generate.eval_diversity import (
    eval_factor_ast_distance,
    eval_pairwise_factor_similarity,
)

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Tuple, List
from tqdm import tqdm

try:
    from colorama import init as colorama_init, Fore, Style

    colorama_init(autoreset=True)
    GREEN = Fore.GREEN
    RED = Fore.RED
    BOLD = Style.BRIGHT
except Exception:
    GREEN = RED = BOLD = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AlphaBench T1 generation pipeline")
    parser.add_argument(
        "--base_dir",
        type=Path,
        default=Path("./runs/T1_Generate_deepseek"),
        help="Base directory to save instructions, outputs, and scores.",
    )
    parser.add_argument(
        "--enable_cot",
        action="store_true",
        help="Enable Chain-of-Thought generation for LLM calls.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-chat",
        help="Model name to use for generation.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="ERROR",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def generate_instruction(origin_instruction):
    instruction = f"""
    You are a financial quantitative researcher.
    Please write a valid factor equivalent to the following description:
    "{origin_instruction}"
    """
    return instruction


def generate_instruction_direction(
    origin_instruction, refer_tags="./benchmark/data/generate/tags_system.json"
):
    with open(refer_tags, "r") as f:
        tags = json.load(f)

    tags_context = ""
    for tag in origin_instruction:
        if tag in tags:
            tags_context += f"{tag}: {tags[tag]}\n"

    instruction = f"""
    You are a financial quantitative researcher.
    Please write a valid factor equivalent meets this direction (we provide the details of each direction tag):
    "{tags_context}"
    """
    return instruction


def build_instructions(
    data_path="./benchmark/data/generate", save_dir="./runs/T1_Generate/instructions"
):

    # Build instruction for zero-shot generation base case
    with open(os.path.join(data_path, "T1_Text2Alpha.json"), "rb") as f:
        data = json.load(f)
        levels = data.keys()
        for level in levels:
            instructions = data[level]
            instructions = [generate_instruction(ins) for ins in instructions]

            with open(
                os.path.join(save_dir, f"T1_1_{level}_instruction.pkl"), "wb"
            ) as f_out:
                pickle.dump(instructions, f_out)

    with open(os.path.join(data_path, "T1_DirectionalMining.json"), "rb") as f:
        data = json.load(f)
        levels = data.keys()
        for level in levels:
            instructions = data[level]
            instructions = [generate_instruction_direction(ins) for ins in instructions]
            with open(
                os.path.join(save_dir, f"T1_2_{level}_instruction.pkl"), "wb"
            ) as f_out:
                pickle.dump(instructions, f_out)

    # Build instruction for instructed generation addition case
    with open(os.path.join(data_path, "T1_Text2Alpha_stability.json"), "rb") as f:
        data = json.load(f)
        chunk_size = len(data)
        for i in range(chunk_size):
            instructions = data[i]["synonym_prompts"]
            instructions = [generate_instruction(ins) for ins in instructions]
            with open(os.path.join(save_dir, f"T1_stability_{i}.pkl"), "wb") as f_out:
                pickle.dump(instructions, f_out)

    # Copy information
    shutil.copy(
        os.path.join(data_path, "T1_DirectionalMining_creativity.json"), save_dir
    )
    shutil.copy(os.path.join(data_path, "T1_Text2Alpha_stability.json"), save_dir)


def load_level_instructions(data_dir: str, prefix: str = "T1_1") -> dict:
    """
    Load every *_{level}_instruction.pkl file that starts with `prefix`.

    Args:
        data_dir: folder that holds the pickle files
        prefix:   file prefix (e.g. "T1_1", "T1_2")

    Returns:
        dict mapping level -> instructions
    """
    loaded = {}
    pattern = f"{prefix}_*_instruction.pkl"  # e.g. "T1_1_*_instruction.pkl"

    for pkl_path in Path(data_dir).glob(pattern):
        # Extract the level name sandwiched between prefix and "instruction"
        # Example: T1_1_easy_instruction.pkl  ->  level = "easy"
        level = pkl_path.stem.split("_")[2]

        with pkl_path.open("rb") as f:
            loaded[level] = pickle.load(f)

    return loaded


def _process_single_idx(idx: int, save_dir: str) -> Tuple[int, Dict[str, Any], str]:
    generate_chunk_path = os.path.join(save_dir, f"T1_creative_{idx}_results.pkl")

    try:
        with open(generate_chunk_path, "rb") as f:
            generate_results = pickle.load(f)
    except Exception as e:
        return idx, {}, f"ERR:{type(e).__name__}"

    collected_factors: List[str] = []
    for item in generate_results:
        if item.get("success") is True:
            try:
                expr = apply_parameters_to_template(
                    item["content"]["expression"]["template"],
                    item["content"]["expression"]["parameters"],
                )
                collected_factors.append(expr)
            except Exception:
                continue

    if not collected_factors:
        return idx, {}, "EMPTY"

    try:
        parser = FactorParser()
        dist = eval_factor_ast_distance(collected_factors, parser)
        corr = eval_pairwise_factor_similarity(collected_factors)
        return idx, {"dist": dist, "corr": corr}, "OK"
    except Exception as e:
        return idx, {}, f"ERR:{type(e).__name__}"


def run_creativity_eval(
    data_path: str, save_dir: str, N_SIZE: int = 30, num_workers: int = 8
) -> Dict[int, Dict[str, Any]]:
    creativity_json = os.path.join(data_path, "T1_DirectionalMining_creativity.json")
    with open(creativity_json, "rb") as f:
        _ = json.load(f)  # just to mirror your original code

    creativity_results: Dict[int, Dict[str, Any]] = {}

    print("Check creative generation for directional mining")

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = {
            ex.submit(_process_single_idx, idx, save_dir): idx for idx in range(N_SIZE)
        }

        for fut in tqdm(
            as_completed(futures), total=N_SIZE, desc="Processing", ncols=80
        ):
            idx = futures[fut]
            try:
                i, result, status = fut.result()
                if result:
                    creativity_results[i] = result
            except Exception as e:
                # catch catastrophic errors
                pass

    print(f"Finished. OK={len(creativity_results)}, TOTAL={N_SIZE}")
    return creativity_results


def start_running_LLM_generation(
    num_workers=8,
    data_path="./runs/T1_Generate/instructions",
    save_dir="./runs/T1_Generate/outputs",
    model="deepseek-chat",
    enable_cot=True,
    local_model=False,
    local_port=8000,
):

    os.makedirs(save_dir, exist_ok=True)

    print("Starting LLM generation current: Text2Alpha")

    text2alpha_instr = load_level_instructions(
        data_path, prefix="T1_1"
    )  # loads all T1_1_*

    for level, instruction_lists in text2alpha_instr.items():
        print(f"Processing level: {level}")

        outputs = batch_call_gen_qlib_factors(
            instructions=instruction_lists,
            num_workers=num_workers,
            verbose=True,
            model=model,
            enable_cot=enable_cot,
            local=local_model,
            local_port=local_port,
        )
        # Save results
        output_path = os.path.join(save_dir, f"T1_1_{level}_results.pkl")
        with open(output_path, "wb") as f_out:
            pickle.dump(outputs, f_out)

    print("Starting LLM generation current: Directional Mining")
    dirmining_instr = load_level_instructions(
        data_path, prefix="T1_2"
    )  # loads all T1_2_*
    for level, instruction_lists in dirmining_instr.items():
        print(f"Processing level: {level}")

        outputs = batch_call_gen_qlib_factors(
            instructions=instruction_lists,
            num_workers=num_workers,
            verbose=True,
            model=model,
            enable_cot=enable_cot,
            local=local_model,
            local_port=local_port,
        )
        # Save results
        output_path = os.path.join(save_dir, f"T1_2_{level}_results.pkl")
        with open(output_path, "wb") as f_out:
            pickle.dump(outputs, f_out)

    print("Starting LLM generation current: Stability")
    pattern = "T1_stability_*.pkl"

    for pkl_path in Path(data_path).glob(pattern):
        # Extract the index number from file name
        # Example: T1_stability_3.pkl -> 3
        idx = int(pkl_path.stem.split("_")[-1])
        with pkl_path.open("rb") as f:
            instructions = pickle.load(f)

        print(f"Processing stability instructions: {idx}")
        outputs = batch_call_gen_qlib_factors(
            instructions=instructions,
            num_workers=num_workers,
            verbose=True,
            model=model,
            enable_cot=enable_cot,
            local=local_model,
            local_port=local_port,
        )

        # Save results
        output_path = os.path.join(save_dir, f"T1_stability_{idx}_results.pkl")
        with open(output_path, "wb") as f_out:
            pickle.dump(outputs, f_out)

    print("Starting LLM generation current: Creative Generation")
    selected_instruction = os.path.join(
        data_path, "T1_DirectionalMining_creativity.json"
    )
    REPEAT_NUM = 10
    dirmining_instr = load_level_instructions(
        data_path, prefix="T1_2"
    )  # loads all T1_2_*
    merged_instruction = []
    with open(selected_instruction, "rb") as f:
        data = json.load(f)
        for level, idx_list in data.items():
            instruction = [dirmining_instr[level][idx] for idx in idx_list]
            merged_instruction.extend(instruction)

    for idx in range(len(merged_instruction)):
        instruction = merged_instruction[idx]
        print(f"Processing creative generation instruction: {idx}")

        instructions_with_id = [
            f"[New Generate Task] {i} Instruction: {instruction}"
            for i in range(REPEAT_NUM)
        ]

        outputs = batch_call_gen_qlib_factors(
            instructions=instructions_with_id,
            num_workers=num_workers,
            verbose=True,
            model=model,
            enable_cot=enable_cot,
            local=local_model,
            local_port=local_port,
            temperature=1.75,
        )

        # Save results
        output_path = os.path.join(save_dir, f"T1_creative_{idx}_results.pkl")
        with open(output_path, "wb") as f_out:
            pickle.dump(outputs, f_out)


def collect_benchmark_scores(
    data_path="./runs/T1_Generate/instructions", save_dir="./runs/T1_Generate/outputs"
):
    scores_record = {}
    text2alpha_instr = load_level_instructions(
        data_path, prefix="T1_1"
    )  # loads all T1_1_*
    print("Load text2alpha instructions:", text2alpha_instr.keys())

    scores_record["text2alpha"] = {}
    scores_record["text2alpha"]["success_rate"] = {}
    scores_record["text2alpha"]["success_count"] = {}
    scores_record["text2alpha"]["pass_rate"] = {}

    for level, _ in text2alpha_instr.items():
        print(f"Processing level: {level}")

        # load results
        output_path = os.path.join(save_dir, f"T1_1_{level}_results.pkl")
        with open(output_path, "rb") as f_out:
            text2alpha_results = pickle.load(f_out)

            success_count = sum(
                1 for item in text2alpha_results if item.get("success") is True
            )
            print(
                f"Success count for {level}: {success_count} out of {len(text2alpha_results)}"
            )
            scores_record["text2alpha"]["success_rate"][level] = success_count / len(
                text2alpha_results
            )
            scores_record["text2alpha"]["success_count"][level] = success_count

    dirmining_instr = load_level_instructions(
        data_path, prefix="T1_2"
    )  # loads all T1_2_*
    print("Load Directional Mining instructions:", dirmining_instr.keys())

    scores_record["directional_mining"] = {}
    scores_record["directional_mining"]["success_rate"] = {}
    scores_record["directional_mining"]["success_count"] = {}
    scores_record["directional_mining"]["pass_rate"] = {}
    for level, _ in dirmining_instr.items():
        print(f"Processing level: {level}")

        # load results
        output_path = os.path.join(save_dir, f"T1_2_{level}_results.pkl")
        with open(output_path, "rb") as f_out:
            dirmining_results = pickle.load(f_out)

            success_count = sum(
                1 for item in dirmining_results if item.get("success") is True
            )

            print(
                f"Success count for {level}: {success_count} out of {len(dirmining_results)}"
            )
            scores_record["directional_mining"]["success_rate"][
                level
            ] = success_count / len(dirmining_results)
            scores_record["directional_mining"]["success_count"][level] = success_count

    print("Check the fitness")
    fitness_filename = "eval_fitness_results.pkl"
    fitness_path = os.path.join(save_dir, "scores", fitness_filename)
    if not os.path.exists(fitness_path):
        print("Fitness results not found, starting evaluation...")
        start_eval_fitness(instruction_dir=data_path, outputs_dir=save_dir)

    # import pdb;pdb.set_trace()

    with open(fitness_path, "rb") as f:
        fitness_results = pickle.load(f)

    for key, results in fitness_results.items():
        _, task_prefix, level = key.split("_")  # e.g., "T1", "1_easy"

        if task_prefix == "1":
            task_key = "text2alpha"
        elif task_prefix == "2":
            task_key = "directional_mining"
        else:
            continue  # skip if unknown

        # import pdb;pdb.set_trace()

        success_rate = (
            results.count("correct") / scores_record[task_key]["success_count"][level]
        )
        print(f"Success rate for {task_key} {level}: {success_rate:.2%}")
        scores_record[task_key]["pass_rate"][level] = success_rate

    creativity_results = {}

    print("Check creative generation for directional mining")
    with open(
        os.path.join(data_path, "T1_DirectionalMining_creativity.json"), "rb"
    ) as f:
        creativity_data = json.load(f)

    N_SIZE = 30

    creativity_results = run_creativity_eval(
        data_path, save_dir, N_SIZE=N_SIZE, num_workers=8
    )
    scores_record["creativity"] = creativity_results

    print("Check stability for Text2Alpha")
    with open(os.path.join(data_path, "T1_Text2Alpha_stability.json"), "rb") as f:
        stability_data = json.load(f)

    similarity_results = {}
    similarity_results["raw"] = {}
    similarity_results["avg"] = {}
    for idx, item in enumerate(stability_data):
        generate_chunk_path = f"{save_dir}/T1_stability_{idx}_results.pkl"
        with open(generate_chunk_path, "rb") as f:
            generate_results = pickle.load(f)

        factor_gt = item["factor_gt"]
        collected_factors = [
            apply_parameters_to_template(
                item["content"]["expression"]["template"],
                item["content"]["expression"]["parameters"],
            )
            for item in generate_results
            if item.get("success") is True
        ]

        similarity = [
            similarity_factor_output(expr, factor_gt) for expr in collected_factors
        ]
        similarity_results["raw"][idx] = similarity
        similarity_results["avg"][idx] = (
            sum(similarity) / len(similarity) if similarity else 0.0
        )
        print(
            f"Checkpoint {idx} similarity for stability: {similarity_results['avg'][idx]}"
        )

    scores_record["stability"] = similarity_results

    # print("Benchmark scores:", scores_record)
    with open(f"{save_dir}/scores/benchmark_scores.pkl", "wb") as f:
        pickle.dump(scores_record, f)
        # json.dump(scores_record, f)


def compute_benchmark_scores(
    benchmark_scores, save_dir, hard_weight=0.4, medium_weight=0.3, easy_weight=0.3
):

    stability_scores = np.mean(list(benchmark_scores["stability"]["avg"].values()))

    creativity_scores_list = []
    for idx, creativity in benchmark_scores["creativity"].items():
        dist_score = creativity["dist"]["diversity"]
        corr_score = creativity["corr"]["diversity"]
        creativity_scores = (dist_score + corr_score) / 2.0
        creativity_scores_list.append(creativity_scores)

    filtered_scores = [x for x in creativity_scores_list if not np.isnan(x)]
    creativity_scores = np.mean(filtered_scores)

    text2alpha_reliability_scores = (
        easy_weight * benchmark_scores["text2alpha"]["success_rate"]["easy"]
        + medium_weight * benchmark_scores["text2alpha"]["success_rate"]["medium"]
        + hard_weight * benchmark_scores["text2alpha"]["success_rate"]["hard"]
    )
    directional_mining_reliability_scores = (
        easy_weight * benchmark_scores["directional_mining"]["success_rate"]["easy"]
        + medium_weight
        * benchmark_scores["directional_mining"]["success_rate"]["medium"]
        + hard_weight * benchmark_scores["directional_mining"]["success_rate"]["hard"]
    )

    text2alpha_pass_scores = (
        0.3 * benchmark_scores["text2alpha"]["pass_rate"]["easy"]
        + 0.3 * benchmark_scores["text2alpha"]["pass_rate"]["medium"]
        + 0.4 * benchmark_scores["text2alpha"]["pass_rate"]["hard"]
    )
    directional_mining_pass_scores = (
        0.3 * benchmark_scores["directional_mining"]["pass_rate"]["easy"]
        + 0.3 * benchmark_scores["directional_mining"]["pass_rate"]["medium"]
        + 0.4 * benchmark_scores["directional_mining"]["pass_rate"]["hard"]
    )

    performance_df = pd.DataFrame(
        {
            "stability": [stability_scores],
            "creativity": [creativity_scores],
            "text2alpha_reliability": [text2alpha_reliability_scores],
            "directional_mining_reliability": [directional_mining_reliability_scores],
            "text2alpha_pass": [text2alpha_pass_scores],
            "directional_mining_pass": [directional_mining_pass_scores],
        }
    )

    detail_performance_df = pd.DataFrame(
        {
            "text2alpha_success_rate_easy": [
                benchmark_scores["text2alpha"]["success_rate"]["easy"]
            ],
            "text2alpha_success_rate_medium": [
                benchmark_scores["text2alpha"]["success_rate"]["medium"]
            ],
            "text2alpha_success_rate_hard": [
                benchmark_scores["text2alpha"]["success_rate"]["hard"]
            ],
            "directional_mining_success_rate_easy": [
                benchmark_scores["directional_mining"]["success_rate"]["easy"]
            ],
            "directional_mining_success_rate_medium": [
                benchmark_scores["directional_mining"]["success_rate"]["medium"]
            ],
            "directional_mining_success_rate_hard": [
                benchmark_scores["directional_mining"]["success_rate"]["hard"]
            ],
        }
    )
    performance_df.to_csv(os.path.join(save_dir, "benchmark_scores.csv"), index=False)
    detail_performance_df.to_csv(
        os.path.join(save_dir, "detail_benchmark_scores.csv"), index=False
    )


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.log_level)

    base_dir: Path = args.base_dir
    instr_dir = base_dir / "instructions"
    out_dir = base_dir / "outputs"
    scores_dir = out_dir / "scores"
    scores_pkl = scores_dir / "benchmark_scores.pkl"

    # Ensure directories exist
    for p in [instr_dir, out_dir, scores_dir]:
        p.mkdir(parents=True, exist_ok=True)

    try:
        # 1) Build instructions
        logging.info("Building instructions → %s", instr_dir)
        build_instructions(
            data_path=str(Path("./benchmark/data/generate").resolve()),
            save_dir=str(instr_dir.resolve()),
        )
        print(GREEN + BOLD + "✔ Instructions generated and saved.")

        # 2) Run LLM generation
        logging.info(
            "Starting LLM generation (model=%s, enable_cot=%s)",
            args.model,
            args.enable_cot,
        )
        start_running_LLM_generation(
            enable_cot=args.enable_cot,
            data_path=str(instr_dir.resolve()),
            save_dir=str(out_dir.resolve()),
            model=args.model,
        )

        # 3) Collect raw benchmark scores
        logging.info("Collecting benchmark scores → %s", out_dir)
        collect_benchmark_scores(
            data_path=str(instr_dir.resolve()), save_dir=str(out_dir.resolve())
        )

        # 4) Load & compute final metrics
        if not scores_pkl.exists():
            logging.error("Expected scores file not found: %s", scores_pkl)

        with scores_pkl.open("rb") as f:
            benchmark_scores = pickle.load(f)

        logging.info("Computing final benchmark metrics → %s", scores_dir)
        compute_benchmark_scores(benchmark_scores, save_dir=str(scores_dir.resolve()))
        print(GREEN + BOLD + "✔ Benchmark metrics computed.")

        logging.info("Done. Outputs in: %s", base_dir.resolve())
        sys.exit(0)

    except Exception as e:
        logging.exception(RED + f"Pipeline failed: {e}")
