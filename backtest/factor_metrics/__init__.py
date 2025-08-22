from .metrics import (
    FL_Ic,
    FL_RankIc,
    FL_Ir,
    FL_Icir,
    FL_RankIcir,
    FL_QuantileReturn,
    FL_Turnover,
)

# Registry of available metrics, user can extend
METRIC_REGISTRY = {
    "ic": FL_Ic,
    "rank_ic": FL_RankIc,
    "ir": FL_Ir,
    "icir": FL_Icir,
    "rank_icir": FL_RankIcir,
    "quantile_return": FL_QuantileReturn,
    "turnover": FL_Turnover,
}


def get_performance(factor_values, label_values, metric_list=None, **kwargs):
    """
    Compute specified metrics for a factor.

    :param factor_values: pd.Series or pd.DataFrame
    :param label_values: pd.Series
    :param metric_list: list of metric names
    :param kwargs: additional args passed to metric functions
    :return: dict of metric results
    """
    if metric_list is None:
        metric_list = list(METRIC_REGISTRY.keys())

    results = {}
    for name in metric_list:
        func = METRIC_REGISTRY.get(name)
        if func is None:
            raise ValueError(f"Unknown metric: {name}")
        try:
            results[name] = func(factor_values, label_values, **kwargs)
        except Exception as e:
            print(f"[Warning] Failed to compute {name}: {e}")
            results[name] = 0
    return results
