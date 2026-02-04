from datetime import datetime
import yaml
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run AlphaBench benchmarks")
    parser.add_argument(
        "--config",
        type=str,
        default="./config/benchmark.yaml",
        help="Path to the configuration file",
    )
    return parser.parse_args()


def run_benchmark(config, result_dir):
    from benchmark.engine import (
        # S1
        run_factor_rebuild_benchmark,
        run_factor_generate_reliability_benchmark,
        # S2
        test_creative_new_factor,
        test_improve_exist_factor,
        # S3
        run_factor_stability_benchmark,
        # S4
        test_factor_performance_improve,
        # S5
        test_control_generate_mcts,
        test_control_generate_ea,
    )

    llm_backend = config.get("BACKEND_LLM", "gpt-4.1-mini")

    local_llm = config.get("BACKEND_SERVICE", "online") == "local"
    llm_port = config.get("BACKEND_LLM_PORT", 8000)

    enable_cot = config.get("ENABLE_COT", True)

    print("Running AlphaBench...")

    # Step 1: Reliability
    s1_cfg = config.get("S1_EVAL_RELIABILITY", {})
    if s1_cfg.get("enable", False):
        print("Running Reliability Benchmark...")
        run_factor_rebuild_benchmark(
            model=llm_backend,
            enable_cot=enable_cot,
            save_dir=result_dir,
            local_model=local_llm,
            local_port=llm_port,
        )
        run_factor_generate_reliability_benchmark(
            model=llm_backend,
            enable_cot=enable_cot,
            save_dir=result_dir,
            local_model=local_llm,
            local_port=llm_port,
        )

    # Step 2: Creativity
    s2_cfg = config.get("S2_EVAL_CREATIVITY", {})
    if s2_cfg.get("enable", False):
        print("Running Creativity Benchmark...")
        test_creative_new_factor(
            model=llm_backend,
            enable_cot=enable_cot,
            save_dir=result_dir,
            local_model=local_llm,
            local_port=llm_port,
        )
        test_improve_exist_factor(
            model=llm_backend,
            enable_cot=enable_cot,
            save_dir=result_dir,
            local_model=local_llm,
            local_port=llm_port,
        )

    # Step 3: Stability
    s3_cfg = config.get("S3_EVAL_STABILITY", {})
    if s3_cfg.get("enable", False):
        print("Running Stability Benchmark...")
        test_data_path = s3_cfg.get("test_data")
        run_factor_stability_benchmark(
            test_data_path,
            save_dir=result_dir,
            model=llm_backend,
            enable_cot=enable_cot,
            local_model=local_llm,
            local_port=llm_port,
        )

    # Step 4: Effectiveness
    # Todo
    s4_cfg = config.get("S4_EVAL_EFFECTIVENESS", {})
    if s4_cfg.get("enable", False):
        print("Running Effectiveness Benchmark...")
        test_factor_performance_improve()

    # Step 5: Controllability

    s5_cfg = config.get("S5_EVAL_CONTROLLABILITY", {})
    if s5_cfg.get("enable", False):
        print("Running Controllability Benchmark...")

        if s5_cfg.get("use_offline_data", True):
            if s5_cfg.get("mscts_instructions"):
                test_control_generate_mcts(
                    model=llm_backend,
                    local_model=local_llm,
                    local_port=llm_port,
                    save_dir=result_dir,
                    use_offline_instructions=True,
                    offline_instructions=s5_cfg["mcts_instructions"],
                    enable_cot=enable_cot,
                )

            if s5_cfg.get("ea_instructions"):
                test_control_generate_ea(
                    model=llm_backend,
                    local_model=local_llm,
                    local_port=llm_port,
                    save_dir=result_dir,
                    use_offline_instructions=True,
                    offline_instructions=s5_cfg["ea_instructions"],
                    enable_cot=enable_cot,
                )
        else:
            raise NotImplementedError(
                "Online MCTS and EA instructions generation is not implemented yet."
            )

    print("All selected benchmarks completed.")


if __name__ == "__main__":
    args = parse_args()
    config_path = args.config

    # Load the configuration file
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Print the whole config dictionary
    print("Parsed Config:")
    print(config)

    llm_backend = config.get("BACKEND_LLM", "gpt-4.1-mini")

    llm_type = config.get("BACKEND_SERVICE", "online")
    llm_port = config.get("BACKEND_LLM_PORT", 8000)

    result_save_path = config.get("RESULT_SAVE_PATH", "./run")
    qlib_data_path = config.get("QLIB_DATA_PATH", "~/.qlib/qlib_data/cn_data")

    # Set it into os.environ so it's available globally
    os.environ["QLIB_DATA_PATH"] = qlib_data_path
    # Accessing individual fields
    print("\nQLIB_DATA_PATH:", qlib_data_path)
    print("RESULT_SAVE_PATH:", result_save_path)
    print("BACKEND_LLM:", llm_backend)

    sub_folder = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(
        result_save_path, llm_type + "_" + llm_backend + "_" + sub_folder
    )
    os.makedirs(result_dir, exist_ok=True)

    # # Run the benchmark
    run_benchmark(config=config, result_dir=result_dir)
