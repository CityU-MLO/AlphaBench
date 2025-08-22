import argparse
import os
import pandas as pd
import yaml
import pickle

import json
from benchmark.utils import (
    check_MCTS_controlled_factor_output,
    check_EA_controlled_factor_output,
)
from factors.lib.alpha158 import load_factors_alpha158
from agent.qlib_contrib.qlib_expr_parsing import FactorParser
from agent.LLMExecutor import FactorGenerationQlibExecutor


def parse_args():
    parser = argparse.ArgumentParser(description="Run AlphaBench benchmarks")
    parser.add_argument(
        "--config",
        type=str,
        default="./config/benchmark.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./runs/benchmark",
        help="Path to the output directory for benchmark results",
    )
    return parser.parse_args()


def generate_radar_plot():
    pass


def generate_S1_scores(output_dir):
    results = {}
    annotation_result_file = os.path.join(
        output_dir, "S1", "factor_annotation_generate_summary.json"
    )
    with open(annotation_result_file, "r") as f:
        annotation_results = json.load(f)

    levels = ["easy", "medium", "hard"]
    case_results = {}
    for case in levels:
        case_result = os.path.join(output_dir, "S1", f"{case}_reliability_summary.json")
        with open(case_result, "r") as f:
            case_results = json.load(f)
        case_results[case] = case_results


def generate_S2_scores(output_dir):
    pass


def generate_S3_scores(output_dir):
    output_result = os.path.join(output_dir, "S3", "stability_summary.csv")
    average_stability = output_result["avg_similarity"]


def generate_S4_scores(output_dir, output_factor_performance):
    # Query from average factor pool performance
    origin_avg_abs_ic = abs(output_factor_performance["origin_IC"]).mean()
    new_avg_abs_ic = abs(output_factor_performance["new_IC"]).mean()
    origin_avg_abs_icir = abs(output_factor_performance["origin_ICIR"]).mean()
    new_avg_abs_icir = abs(output_factor_performance["new_ICIR"]).mean()
    origin_avg_abs_rankic = abs(output_factor_performance["origin_RankIC"]).mean()
    new_avg_abs_rankic = abs(output_factor_performance["new_RankIC"]).mean()

    rel_ic = (new_avg_abs_ic - origin_avg_abs_ic) / abs(origin_avg_abs_ic)
    rel_icir = (new_avg_abs_icir - origin_avg_abs_icir) / abs(origin_avg_abs_icir)
    rel_rankic = (new_avg_abs_rankic - origin_avg_abs_rankic) / abs(
        origin_avg_abs_rankic
    )

    effectiveness_score = max(0, min(1, (rel_ic + rel_icir + rel_rankic) / 3 + 0.5))

    return effectiveness_score


def generate_S5_scores(output_dir):
    result_report = {}
    ## Test for MCTS
    with open(os.path.join(output_dir, "S5", "mcts_factor_result.json"), "r") as f:
        factor_data = json.load(f)

    with open(os.path.join(output_dir, "S5", "mcts_factor_instruction.pkl"), "rb") as f:
        instruction_data = pickle.load(f)

    standard_factors, compile_factors = load_factors_alpha158()

    for factor in standard_factors:
        factor["expression"]["template"] = compile_factors.get(factor["name"], {}).get(
            "qlib_expression"
        )

    valid_count = 0
    for i in range(len(instruction_data)):
        generated_factor = factor_data[i]
        origin_factor = [
            d
            for d in standard_factors
            if d.get("name") == instruction_data[i]["factor_name"]
        ][0]
        status = check_MCTS_controlled_factor_output(
            generated_factor, origin_factor, instruction_data[i], verbose=True
        )
        if status:
            valid_count += 1

    print(
        f"[MCTS] Total valid factors: {valid_count}/{len(factor_data)}, ratio: {valid_count / len(factor_data):.2%}"
    )

    result_report["mcts_valid_rate"] = valid_count / len(factor_data)

    with open(os.path.join(output_dir, "S5", "ea_factor_result.json"), "r") as f:
        factor_data = json.load(f)

    with open(os.path.join(output_dir, "S5", "ea_factor_instruction.pkl"), "rb") as f:
        instruction_data = pickle.load(f)

    instruction_type_list = [instruction["type"] for instruction in instruction_data]
    valid_status_list = []
    for i in range(len(factor_data)):
        generated_factor = factor_data[i]
        instruction = instruction_data[i]
        status = check_EA_controlled_factor_output(
            generated_factor, instruction, verbose=True
        )
        if status:
            valid_status_list.append(1)
        else:
            valid_status_list.append(0)

        print(f"Factor {i} EA control check: {'Passed' if status else 'Failed'}")

    print(
        f"[EA] Total valid factors: {sum(valid_status_list)}/{len(factor_data)}, ratio: {sum(valid_status_list) / len(factor_data):.2%}"
    )

    result_report["ea_valid_rate"] = sum(valid_status_list) / len(factor_data)
    # Compute mutation success rate
    mutation_indices = [
        i for i, t in enumerate(instruction_type_list) if t == "mutation"
    ]
    mutation_successes = [valid_status_list[i] for i in mutation_indices]
    result_report["ea_mutation_success_rate"] = (
        sum(mutation_successes) / len(mutation_successes) if mutation_successes else 0.0
    )

    # Compute crossover success rate
    crossover_indices = [
        i for i, t in enumerate(instruction_type_list) if t == "crossover"
    ]
    crossover_successes = [valid_status_list[i] for i in crossover_indices]
    result_report["ea_crossover_success_rate"] = (
        sum(crossover_successes) / len(crossover_successes)
        if crossover_successes
        else 0.0
    )

    result_report_df = pd.DataFrame([result_report])
    result_report_file = os.path.join(output_dir, "S5_result_report.csv")
    result_report_df.to_csv(result_report_file, index=False)


def collect_benchmark_results(config, result_dir):

    llm_backend = config.get("BACKEND_LLM", "gpt-4.1-mini")

    local_llm = config.get("BACKEND_SERVICE", "online") == "local"
    llm_port = config.get("BACKEND_LLM_PORT", 8000)

    enable_cot = config.get("ENABLE_COT", True)

    # Step 1: Reliability
    s1_cfg = config.get("S1_EVAL_RELIABILITY", {})
    if s1_cfg.get("enable", False):
        print("Running Reliability Benchmark...")

    # Step 2: Creativity
    s2_cfg = config.get("S2_EVAL_CREATIVITY", {})
    if s2_cfg.get("enable", False):
        print("Running Creativity Benchmark...")

    # Step 3: Stability
    s3_cfg = config.get("S3_EVAL_STABILITY", {})
    if s3_cfg.get("enable", False):
        print("Running Stability Benchmark...")

    # Step 4: Effectiveness
    s4_cfg = config.get("S4_EVAL_EFFECTIVENESS", {})
    if s4_cfg.get("enable", False):
        print("Running Effectiveness Benchmark...")

    # Step 5: Controllability

    s5_cfg = config.get("S5_EVAL_CONTROLLABILITY", {})
    if s5_cfg.get("enable", False):
        print("Running Controllability Benchmark...")


if __name__ == "__main__":
    print("Generate performance report")
    args = parse_args()
    config_path = args.config
    output_dir = os.makedir()

    # Load the configuration file
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Print the whole config dictionary
    print("Parsed Config:")
    print(config)
