import qlib
import agent.qlib_contrib.qlib_extend_ops
from qlib.constant import REG_CN
from qlib.data.dataset.loader import QlibDataLoader
import pandas as pd
import os

try:
    provider_uri = os.environ.get("QLIB_DATA_PATH", "~/.qlib/qlib_data/cn_data")
    print(f"Using QLIB_DATA_PATH: {provider_uri}")
    qlib.init(provider_uri=provider_uri, region=REG_CN)
    print(f"Qlib initialized with data provider at {provider_uri}")
    QLIB_VALID_FLAG = True
except Exception as e:
    print(f"Error initializing Qlib: {e}")
    QLIB_VALID_FLAG = False
    raise


def compute_factor_data(
    factor_definitions,
    label="close_return",
    instruments="csi300",
    start_time="2021-01-01",
    end_time="2025-01-01",
):
    """
    Compute factor data and label data using QlibDataLoader.

    :param factor_definitions: List of dicts, each with 'name' and 'expression'
                               e.g. [{'name': 'factor1', 'expression': 'Rank(Div($high, $low), 10)'}]
    :param label: String, label expression, e.g. 'Ref($close, -2)/Ref($close, -1) - 1'
    :param instruments: String, instruments universe, default 'csi300'
    :param start_time: String, start date
    :param end_time: String, end date
    :return: pandas DataFrame containing factor data and label data
    """
    # Extract fields and names from factor_definitions
    fields = [f["expression"] for f in factor_definitions]
    names = [f["name"] for f in factor_definitions]

    # Prepare label
    if label == "close_return_lag":
        label = "Ref($close, -2)/Ref($close, -1) - 1"
    elif label == "close_return":
        label = "Ref($close, -1)/$close - 1"
    elif label == "close":
        label = "Ref($close, -1)"
    else:
        raise ValueError(
            f"Unsupported label type: {label}. Supported types are 'close_return' or 'close'."
        )

    labels = [label]
    label_names = ["LABEL"]

    # Build config
    data_loader_config = {"feature": (fields, names), "label": (labels, label_names)}

    # Load data
    data_loader = QlibDataLoader(config=data_loader_config)
    try:
        df = data_loader.load(
            instruments=instruments, start_time=start_time, end_time=end_time
        )

        if df.empty:
            print("Error: Loaded DataFrame is empty.")
            return None
        return df
    except Exception as e:
        print(f"Exception during factor computation: {e}")
        return None


def main():
    factor_definitions = [
        {
            "name": "vol_momentum",
            "expression": "(Sum(Greater($volume-Ref($volume, 1), 0), 10)-Sum(Greater(Ref($volume, 1)-$volume, 0), 10))/(Sum(Abs($volume-Ref($volume, 1)), 10)+1e-12)",
        },
        {"name": "price_ratio", "expression": "Rank(Div($high, $low), 10)"},
    ]

    label = "close_return"

    df = compute_factor_data(
        factor_definitions, label, start_time="2020-01-01", end_time="2020-01-03"
    )

    print(df)


if __name__ == "__main__":
    main()
