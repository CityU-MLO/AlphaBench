def evaluate_ranking_precision(ranking_list, ground_truth, N=10):
    """
    Evaluate the top-N precision of a given ranking against the ground truth.

    Args:
        ranking_list (list): Predicted ranking of factor IDs (sorted high to low).
        ground_truth (list): Ground truth ranking of factor IDs (sorted high to low).
        N (int): Number of top items to evaluate (default=10).

    Returns:
        float: Top-N precision score.
    """
    if not ranking_list or not ground_truth:
        return 0.0

    # Take only top-N
    pred_topN = ranking_list[:N]
    gt_topN = set(ground_truth[:N])

    # Count matches
    hits = sum(1 for f in pred_topN if f in gt_topN)
    precision = hits / N

    return precision


def evaluate_scoring_mse(pred_scores, true_scores):
    """
    Evaluate the scoring task using Mean Squared Error (MSE).

    Args:
        pred_scores (dict): Predicted scores {factor_id: int (1–5)}.
        true_scores (dict): Ground truth scores {factor_id: int (1–5)}.

    Returns:
        float: Mean squared error between predicted and ground-truth scores.
    """
    if not pred_scores or not true_scores:
        return 0.0

    errors = []
    for fid, true_s in true_scores.items():
        if fid in pred_scores:
            pred_s = pred_scores[fid]
            errors.append((true_s - pred_s) ** 2)

    if not errors:
        return 0.0

    mse = sum(errors) / len(errors)
    return mse


if __name__ == "__main__":
    # Example test
    predicted = ["f1", "f7", "f3", "f9", "f12", "f20", "f5", "f8", "f15", "f18"]
    ground_truth = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10"]

    score = evaluate_ranking_precision(predicted, ground_truth, N=5)
    print("Top-5 Precision:", score)

    score = evaluate_ranking_precision(predicted, ground_truth, N=10)
    print("Top-10 Precision:", score)

    # Example test
    pred_scores = {"f1": 3, "f2": 5, "f3": 2, "f4": 4}
    true_scores = {"f1": 4, "f2": 5, "f3": 1, "f4": 3}

    mse = evaluate_scoring_mse(pred_scores, true_scores)
    print("Scoring MSE:", mse)
