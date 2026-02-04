import json
from math import nan
import math
import numpy as np
from backtest.qlib.dataloader import compute_factor_data
from backtest.factor_metrics import get_performance
from agent.qlib_contrib.qlib_expr_parsing import FactorParser, print_tree


def similarity_factor_output(factor_expr, truth_expr, tol=1e-2, verbose=False):
    try:
        input_expr = [
            {"name": "generated_factor", "expression": factor_expr},
            {"name": "truth_factor", "expression": truth_expr},
        ]
        df = compute_factor_data(
            input_expr, start_time="2024-01-01", end_time="2024-03-01"
        )

        # Compute absolute difference
        abs_diff = (
            df["feature"]["generated_factor"] - df["feature"]["truth_factor"]
        ).abs()

        # Check if difference is within tolerance everywhere
        all_close = abs_diff.sum()

        if all_close < tol:
            if verbose:
                print(
                    "✅ The generated factor and truth factor outputs are identical (within tolerance)."
                )
            return 1
        else:
            max_diff = abs_diff.max()

            # Compute correlation as a fallback comparison
            corr = df["feature"][["generated_factor", "truth_factor"]].corr().iloc[0, 1]
            if verbose:
                print(f"⚠️ Differences found (max diff: {max_diff:.2e}).")
                print(f"📊 Correlation between generated and truth factor: {corr:.4f}")

            return corr
    except Exception as e:
        return 0


def compare_factor_output(factor_expr_list, tol=1e-2, verbose=False):
    """
    Compare multiple factor outputs for similarity and variance.
    
    Args:
        factor_expr_list (list of str): List of factor """ "expression" """s.
        tol (float): Tolerance for declaring outputs as identical.
        verbose (bool): Whether to print detailed messages.
    
    Returns:
        dict: Summary statistics including:
            - mean_corr: Mean pairwise correlation between factors
            - total_variance: Sum of individual factor variances
            - all_close: Whether all factors are identical within tolerance
    """
    # Prepare input_expr with feat0, feat1, ..., featN
    input_expr = [
        {"name": f"feat{i}", "expression": expr}
        for i, expr in enumerate(factor_expr_list)
    ]

    # Compute factor data (assume compute_factor_data returns DataFrame with "feature" group)
    df = compute_factor_data(input_expr)
    feature_df = df["feature"]

    # Compute absolute differences between all pairs
    all_close = True
    num_feats = len(feature_df.columns)
    for i in range(num_feats):
        for j in range(i + 1, num_feats):
            col_i = feature_df.columns[i]
            col_j = feature_df.columns[j]
            abs_diff = (feature_df[col_i] - feature_df[col_j]).abs()
            diff_sum = abs_diff.sum()
            if diff_sum >= tol:
                all_close = False
                if verbose:
                    max_diff = abs_diff.max()
                    print(
                        f"⚠️ Difference between {col_i} and {col_j} (max diff: {max_diff:.2e})"
                    )

    # Compute total variance
    total_variance = feature_df.var().sum()

    # Compute mean pairwise correlation
    if num_feats >= 2:
        corr_matrix = feature_df.corr()
        upper_tri_indices = np.triu_indices_from(corr_matrix, k=1)
        corrs = corr_matrix.values[upper_tri_indices]
        mean_corr = np.nanmean(corrs) if len(corrs) > 0 else np.nan
    else:
        mean_corr = 1.0  # Only one factor — consider correlation perfect

    if verbose:
        print(f"📊 Mean correlation between factors: {mean_corr:.4f}")
        print(f"📈 Total variance sum: {total_variance:.4f}")
        if all_close:
            print("✅ All factor outputs are identical (within tolerance).")
        else:
            print("❌ Factor outputs differ beyond tolerance.")

    return {
        "mean_corr": mean_corr,
        "total_variance": total_variance,
        "all_close": all_close,
    }


def get_factor_performance_batch(
    factor_name_expr_lists, start_time="2021-01-01", end_time="2025-01-01"
):
    input_expr = [
        {"name": name, "expression": expr} for name, expr in factor_name_expr_lists
    ]

    df = compute_factor_data(
        input_expr, label="close_return", start_time=start_time, end_time=end_time
    )
    features = df["feature"]
    label = df["label"]["LABEL"]

    performance_dict = {}
    for name in features.columns:
        factor_data = features[name]
        if factor_data.empty or label.empty:
            print(f"Warning: Factor {name} or label data is empty.")
            performance_dict[name] = {
                "IC": 0,
                "ICIR": 0,
                "RankIC": 0,
                "RankICIR": 0,
                "QuantileReturn": 0,
            }
            continue

        # Compute performance metrics
        perf = get_performance(
            factor_data, label, metric_list=["ic", "icir", "rank_ic", "rank_icir"]
        )
        # import pdb;pdb.set_trace()
        performance_dict[name] = {
            "IC": round(float(perf.get("ic", 0)), 4),
            "ICIR": round(float(perf.get("icir", 0)), 4),
            "RankIC": round(float(perf.get("rank_ic", 0)), 4),
            "RankICIR": round(float(perf.get("rank_icir", 0)), 4),
            "QuantileReturn": round(float(perf.get("quantile_return", 0)), 4),
        }

    return performance_dict


def compute_factor_diversity(factor_expr_list):
    """
    Compute the structural diversity of a list of factor """ "expression" """s based on their 
    parsed syntax trees.

    The function parses each """ "expression" """ into a tree structure and computes pairwise 
    tree edit distances (using zss.simple_distance). It reports:
        - The mean distance between all valid pairs
        - The maximum distance between all valid pairs
        - A diversity score defined as mean_dist / max_dist (0 if max_dist is 0)

    Any """ "expression" """ that fails to parse will be skipped with a warning.

    Parameters
    ----------
    factor_expr_list : list of str
        A list of factor """ "expression" """s (e.g., formula strings) to evaluate.

    Returns
    -------
    tuple
        (mean_dist, max_dist, diversity_score)
        - mean_dist : float
            The average pairwise tree distance.
        - max_dist : float
            The largest pairwise tree distance.
        - diversity_score : float
            A normalized diversity score (mean_dist / max_dist).
            Returns 0 if max_dist is 0.
    """
    parser = FactorParser()  # or your parser
    trees = []

    for expr in factor_expr_list:
        try:
            tree = parser.parse(expr)
            trees.append(tree)
        except Exception as e:
            print(f"⚠️ Warning: Failed to parse " """expression""" ": {expr}")
            print(f"   Reason: {e}")
            continue

    N = len(trees)
    if N < 2:
        print("Warning: Not enough valid trees to compute diversity. Returning 0.")
        return 0, 0, 0  # mean, max, score

    from zss import simple_distance

    pairwise = []

    for i in range(N):
        for j in range(i + 1, N):
            try:
                dist = simple_distance(
                    trees[i],
                    trees[j],
                    get_children=lambda node: node.children,
                    get_label=lambda node: node.label,
                )
                pairwise.append(dist)
            except Exception as e:
                print(
                    f"Warning: Failed to compute distance between tree {i} and tree {j}"
                )
                print(f"   Reason: {e}")

    if not pairwise:
        print("Warning: No distances computed successfully. Returning 0.")
        return 0, 0, 0

    mean_dist = sum(pairwise) / len(pairwise)
    max_dist = max(pairwise)

    # Compute diversity score
    if max_dist > 0:
        diversity_score = mean_dist / max_dist
    else:
        diversity_score = 0

    return mean_dist, max_dist, diversity_score


if __name__ == "__main__":
    # Example usage
    with open("./factors/lib/alpha158/qlib_compile_product.json", "r") as f:
        generated_factors = json.load(f)

    facs = [fac.get("qlib_" """expression""" "_default") for fac in generated_factors]
    mean_dist, max_dist, diversity_score = compute_factor_diversity(facs)
    print(
        f"Mean Distance: {mean_dist:.4f}, Max Distance: {max_dist:.4f}, Diversity Score: {diversity_score:.4f}"
    )

    performance_fac = get_factor_performance_batch(
        [
            (fac.get("name"), fac.get("qlib_" """expression""" "_default"))
            for fac in generated_factors
        ]
    )

    json.dump(
        performance_fac, open("performance_fac.json", "w"), indent=2, ensure_ascii=False
    )

    # Define factor definitions
    # This should match the structure expected by compute_factor_data
    # test_data = {}
    # for factor in generated_factors[:5]:
    #     name = factor.get("name")
    #     expr = factor.get("qlib_""""expression""""_default")

    #     test_data[name] = {"generated": expr, "truth": expr}

    # # Test factor list inter-comparison
    # factor_diff = compare_factor_output(
    #     [fac.get("qlib_""""expression""""_default") for fac in generated_factors], verbose=True
    # )
    # print(f"Factor comparison results: {factor_diff}")

    # # Test factor output comparison
    # for name, data in test_data.items():
    #     generated_expr = data["generated"]
    #     truth_expr = data["truth"]
    #     similarity = similarity_factor_output(generated_expr, truth_expr, verbose=True)

    #     print(f"Factor: {name}, Similarity: {similarity:.4f}")
