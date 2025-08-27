import json
import logging
import os
import pickle
import sys
import numpy as np
import pandas as pd

from benchmark.engine.generate.benchmark_main import load_level_instructions, run_creativity_eval
from benchmark.engine.generate.eval_fitness import start_eval_fitness
from benchmark.engine.utils import similarity_factor_output

def compute_benchmark_scores(
    benchmark_scores, save_dir, hard_weight=0.4, medium_weight=0.3, easy_weight=0.3
):

    stability_scores = np.mean(list(benchmark_scores["stability"]["avg"].values()))

    # creativity_scores_list = []
    # for idx, creativity in benchmark_scores["creativity"].items():
    #     dist_score = creativity["dist"]["diversity"]
    #     corr_score = creativity["corr"]["diversity"]
    #     creativity_scores = (dist_score + corr_score) / 2.0
    #     creativity_scores_list.append(creativity_scores)

    # filtered_scores = [x for x in creativity_scores_list if not np.isnan(x)]
    # creativity_scores = np.mean(filtered_scores)
    creativity_scores = 0

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

    text2alpha_pass_scores = 0
    directional_mining_pass_scores = 0
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
    N_SIZE = 30

    # creativity_results = run_creativity_eval(
    #     data_path, save_dir, N_SIZE=N_SIZE, num_workers=8
    # )
    # scores_record["creativity"] = creativity_results

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
            item["content"]["expression"]
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



def collect_main(instr_dir, out_dir):
    # 1) Collect raw benchmark scores
    logging.info("Collecting benchmark scores → %s", out_dir)
    scores_dir = os.path.join(out_dir, "scores")
    os.makedirs(scores_dir, exist_ok=True)
    
    collect_benchmark_scores(
        data_path=instr_dir, save_dir=out_dir)
    
    scores_pkl = os.path.join(scores_dir, "benchmark_scores.pkl")
    
    # 2) Load & compute final metrics

    with open(scores_pkl, "rb") as f:
        benchmark_scores = pickle.load(f)

    logging.info("Computing final benchmark metrics → %s", scores_dir)
    compute_benchmark_scores(benchmark_scores, save_dir=scores_dir)
    print( "✔ Benchmark metrics computed.")

    logging.info("Done. Outputs in: %s", scores_pkl)
    
if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--input_dir",
    #     type=str,
    #     required=True,
    #     help="Directory containing the generated results to be collected.",
    # )
    # args = parser.parse_args()
    
    # instruction_dir = os.path.join(args.input_dir, "instructions")
    # output_dir = os.path.join(args.input_dir, "outputs")
    # collect_main(instruction_dir, output_dir)
    base_dir = '/home/hluo/workdir/AlphaBench/runs/T1_official'
    models = [
        # 'gpt-5_cot_false',
        'gemini-2.5-pro_cot_false',
        'deepseek-ai_DeepSeek-V3_cot_false',
        'deepseek-ai_DeepSeek-V3_cot_true',
        'gpt-4.1-mini_cot_false',
        'gpt-4.1-mini_cot_true',
        'gemini-2.5-flash_cot_false',
        'gemini-2.5-flash_cot_true',
        'llama-3.1-70b-instruct_cot_false',
        'llama-3.1-70b-instruct_cot_true',     
    ]
    for model in models:
        print("Processing:", model)
        collect_main(os.path.join(base_dir, model, "instructions"), os.path.join(base_dir, model, "outputs"))