import json
from agent.generator_qlib import call_gen_qlib_factors
from agent.compiler import apply_parameters_to_template

# from benchmutils import *
from benchmark.engine.utils import similarity_factor_output
import numpy as np
from itertools import combinations
from typing import List, Dict, Any, Tuple, List

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed


def eval_factor_ast_distance(
    factors: List[str], parser: Any = None, return_matrix: bool = True
) -> Dict[str, Any]:
    """
    Evaluate structural distances among factor expressions via AST tree-edit distance.

    Uses zss.simple_distance on parsed trees.
    - mean_dist: mean pairwise tree-edit distance (ignoring failed / NaN pairs)
    - max_dist: max pairwise distance
    - diversity: normalized diversity = mean_dist / max_dist (0 if max_dist == 0)
    - n_pairs: number of valid pairs used
    - dist_pairs: list of ((i, j), dist) for i<j
    - dist_matrix (optional): N x N symmetric matrix (0 on diagonal), NaN on failures

    Args:
        factors: list of factor expression strings.
        parser: optional parser instance. If None, will try to use global `FactorParser()`.
        return_matrix: whether to include the full distance matrix.

    Returns:
        Dict[str, Any]
    """
    # 1) Prepare parser
    if parser is None:
        ParserClass = globals().get("FactorParser", None)
        if ParserClass is None:
            raise ValueError(
                "No parser provided and `FactorParser` not found in globals()."
            )
        parser = ParserClass()

    # 2) Parse expressions to trees
    trees: List[Any] = []
    ok_indices: List[int] = []
    for i, expr in enumerate(factors):
        try:
            t = parser.parse(expr)
            trees.append(t)
            ok_indices.append(i)
        except Exception as e:
            # Skip with silent failure per your reference (you can log if needed)
            trees.append(None)

    n = len(factors)
    if n == 0:
        return {
            "mean_dist": np.nan,
            "max_dist": np.nan,
            "diversity": np.nan,
            "n_pairs": 0,
            "dist_pairs": [],
            **({"dist_matrix": np.empty((0, 0))} if return_matrix else {}),
        }

    # 3) Pairwise tree-edit distances
    try:
        from zss import simple_distance
    except Exception as e:
        raise ImportError("zss is required: pip install zss") from e

    dist_mat = np.full((n, n), np.nan, dtype=float)
    np.fill_diagonal(dist_mat, 0.0)

    dist_pairs: List[Tuple[Tuple[int, int], float]] = []
    for i in range(n):
        ti = trees[i]
        if ti is None:
            continue
        for j in range(i + 1, n):
            tj = trees[j]
            if tj is None:
                continue
            try:
                d = simple_distance(
                    ti,
                    tj,
                    get_children=lambda node: node.children,
                    get_label=lambda node: node.label,
                )
            except Exception:
                d = np.nan
            dist_mat[i, j] = dist_mat[j, i] = d
            dist_pairs.append(((i, j), d))

    # 4) Aggregate stats
    vals = [d for _, d in dist_pairs if d == d]  # filter NaNs
    mean_dist = float(np.mean(vals)) if vals else 0.0
    max_dist = float(np.max(vals)) if vals else 0.0
    diversity = (mean_dist / max_dist) if max_dist > 0 else 0.0

    out: Dict[str, Any] = {
        "mean_dist": mean_dist,
        "max_dist": max_dist,
        "diversity": diversity,
        "n_pairs": len(vals),
        "dist_pairs": dist_pairs,
    }
    if return_matrix:
        out["dist_matrix"] = dist_mat
    return out


def eval_pairwise_factor_similarity(
    factors: List[str],
    return_matrix: bool = True,
    *,
    num_workers: int = 1,  # 0/1 => single-thread; >1 => parallel
    use_processes: bool = False,  # True => ProcessPool, else threads
    show_progress: bool = False,  # show tqdm progress bar if available
    store_pairs: bool = True,  # keep corr_pairs (can be large)
) -> Dict[str, Any]:
    """
    Efficient pairwise similarity among factor expressions using ONLY
    similarity_factor_output(f1, f2). No caching.

    Returns dict with:
      - mean_corr, mean_abs_corr, diversity=1-mean_abs_corr
      - n_pairs
      - corr_pairs (optional): [((i,j), corr)]
      - corr_matrix (optional): n×n symmetric with diag=1
    """
    n = len(factors)
    if n == 0:
        out = {
            "mean_corr": np.nan,
            "mean_abs_corr": np.nan,
            "diversity": np.nan,
            "n_pairs": 0,
            "corr_pairs": [] if store_pairs else [],
        }
        if return_matrix:
            out["corr_matrix"] = np.empty((0, 0))
        return out
    if n == 1:
        out = {
            "mean_corr": np.nan,
            "mean_abs_corr": np.nan,
            "diversity": np.nan,
            "n_pairs": 0,
            "corr_pairs": [] if store_pairs else [],
        }
        if return_matrix:
            out["corr_matrix"] = np.array([[1.0]])
        return out

    # 1) Deduplicate factors (identical expressions → auto-corr = 1.0)
    unique_map: Dict[str, int] = {}
    unique_factors: List[str] = []
    groups: List[List[int]] = []  # for each unique id, list original indices
    for idx, f in enumerate(factors):
        if f in unique_map:
            uid = unique_map[f]
            groups[uid].append(idx)
        else:
            uid = len(unique_factors)
            unique_map[f] = uid
            unique_factors.append(f)
            groups.append([idx])

    m = len(unique_factors)

    # Prepare corr matrix
    corr_mat = None
    if return_matrix:
        corr_mat = np.full((n, n), np.nan, dtype=float)
        np.fill_diagonal(corr_mat, 1.0)
        # Fill same-expression blocks with 1.0 directly
        for g in groups:
            for i in range(len(g)):
                for j in range(i + 1, len(g)):
                    ii, jj = g[i], g[j]
                    corr_mat[ii, jj] = corr_mat[jj, ii] = 1.0

    # If all factors identical, short-circuit
    if m == 1:
        total_pairs = n * (n - 1) // 2
        if total_pairs == 0:
            mean_corr = mean_abs_corr = np.nan
            diversity = np.nan
        else:
            mean_corr = 1.0
            mean_abs_corr = 1.0
            diversity = 0.0
        out = {
            "mean_corr": float(mean_corr),
            "mean_abs_corr": float(mean_abs_corr),
            "diversity": float(diversity) if not np.isnan(diversity) else np.nan,
            "n_pairs": total_pairs,
            "corr_pairs": (
                []
                if not store_pairs
                else [((i, j), 1.0) for i in range(n) for j in range(i + 1, n)]
            ),
        }
        if return_matrix:
            out["corr_matrix"] = corr_mat
        return out

    # 2) Unique upper-triangular pairs among unique expressions
    unique_pairs = [(u, v) for u, v in combinations(range(m), 2)]

    def _eval_pair(u: int, v: int) -> Tuple[int, int, float]:
        f1, f2 = unique_factors[u], unique_factors[v]
        try:
            c = similarity_factor_output(f1, f2)  # float in [-1,1] or NaN
        except Exception:
            c = np.nan
        return u, v, c

    # 3) Compute unique pair correlations (parallel optional)
    results_uv: Dict[Tuple[int, int], float] = {}
    if num_workers and num_workers > 1:
        Executor = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        iterator = None
        with Executor(max_workers=num_workers) as ex:
            futures = [ex.submit(_eval_pair, u, v) for (u, v) in unique_pairs]
            iterator = as_completed(futures)
            if show_progress:
                try:
                    from tqdm import tqdm  # type: ignore

                    iterator = tqdm(
                        iterator, total=len(futures), desc="Pairwise corr", ncols=80
                    )
                except Exception:
                    pass
            for fut in iterator:
                u, v, c = fut.result()
                results_uv[(u, v)] = c
    else:
        pairs_iter = unique_pairs
        if show_progress:
            try:
                from tqdm import tqdm  # type: ignore

                pairs_iter = tqdm(
                    pairs_iter, total=len(unique_pairs), desc="Pairwise corr", ncols=80
                )
            except Exception:
                pass
        for u, v in pairs_iter:
            _, _, c = _eval_pair(u, v)
            results_uv[(u, v)] = c

    # 4) Expand unique results to full n×n index space, gather stats
    corr_pairs_list: List[Tuple[Tuple[int, int], float]] = [] if store_pairs else []
    valid_vals: List[float] = []

    # Cross-group fills
    for (u, v), c in results_uv.items():
        idxs_u = groups[u]
        idxs_v = groups[v]
        if return_matrix:
            for ii in idxs_u:
                for jj in idxs_v:
                    corr_mat[ii, jj] = corr_mat[jj, ii] = c
        for ii in idxs_u:
            for jj in idxs_v:
                if not np.isnan(c):
                    valid_vals.append(c)
                if store_pairs:
                    i, j = (ii, jj) if ii < jj else (jj, ii)
                    corr_pairs_list.append(((i, j), c))

    # Intra-group pairs (identical expressions) stats/pairs
    for g in groups:
        if len(g) > 1:
            for i_local in range(len(g)):
                for j_local in range(i_local + 1, len(g)):
                    ii, jj = g[i_local], g[j_local]
                    valid_vals.append(1.0)
                    if store_pairs:
                        corr_pairs_list.append(((ii, jj), 1.0))

    # 5) Summary stats
    if valid_vals:
        mean_corr = float(np.mean(valid_vals))
        mean_abs_corr = float(np.mean(np.abs(valid_vals)))
    else:
        mean_corr = np.nan
        mean_abs_corr = np.nan
    diversity = 1.0 - mean_abs_corr if not np.isnan(mean_abs_corr) else np.nan

    out: Dict[str, Any] = {
        "mean_corr": mean_corr,
        "mean_abs_corr": mean_abs_corr,
        "diversity": diversity,
        "n_pairs": len(valid_vals),
        "corr_pairs": corr_pairs_list if store_pairs else [],
    }
    if return_matrix:
        out["corr_matrix"] = corr_mat
    return out
