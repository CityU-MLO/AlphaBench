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
        default=Path("./runs/T1_deepseek"),
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
        "--local_model",
        action="store_true",
        help="Use a local LLM server for generation.",
    )
    parser.add_argument(
        "--local_port",
        type=int,
        default=8000,
        help="Port number for the local LLM server.",
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
    if isinstance(origin_instruction, str):
        tags_context += f"{origin_instruction}: {tags.get(origin_instruction, '')}\n"
    else:
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
            # import pdb;pdb.set_trace()
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
                expr = item["content"]["expression"]
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
        if level != 'easy':
            continue
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

    # print("Starting LLM generation current: Creative Generation")
    # selected_instruction = os.path.join(
    #     data_path, "T1_DirectionalMining_creativity.json"
    # )
    # REPEAT_NUM = 10
    # dirmining_instr = load_level_instructions(
    #     data_path, prefix="T1_2"
    # )  # loads all T1_2_*
    # merged_instruction = []
    # with open(selected_instruction, "rb") as f:
    #     data = json.load(f)
    #     for level, idx_list in data.items():
    #         instruction = [dirmining_instr[level][idx] for idx in idx_list]
    #         merged_instruction.extend(instruction)

    # for idx in range(len(merged_instruction)):
    #     instruction = merged_instruction[idx]
    #     print(f"Processing creative generation instruction: {idx}")

    #     instructions_with_id = [
    #         f"[New Generate Task] {i} Instruction: {instruction}"
    #         for i in range(REPEAT_NUM)
    #     ]

    #     outputs = batch_call_gen_qlib_factors(
    #         instructions=instructions_with_id,
    #         num_workers=num_workers,
    #         verbose=True,
    #         model=model,
    #         enable_cot=enable_cot,
    #         local=local_model,
    #         local_port=local_port,
    #         temperature=1.75,
    #     )

    #     # Save results
    #     output_path = os.path.join(save_dir, f"T1_creative_{idx}_results.pkl")
    #     with open(output_path, "wb") as f_out:
    #         pickle.dump(outputs, f_out)



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
            num_workers=4,
            enable_cot=args.enable_cot,
            data_path=str(instr_dir.resolve()),
            save_dir=str(out_dir.resolve()),
            model=args.model,
            local_model=args.local_model,
            local_port=args.local_port,
        )


        sys.exit(0)

    except Exception as e:
        logging.exception(RED + f"Pipeline failed: {e}")
