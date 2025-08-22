import qlib
import agent.qlib_contrib.qlib_extend_ops
from qlib.constant import REG_CN
from qlib.data.dataset.loader import QlibDataLoader
import pandas as pd
import threading


def _load_data_thread(q, config):
    try:
        data_loader = QlibDataLoader(config=config)
        df = data_loader.load(
            instruments="csi300", start_time="2020-01-01", end_time="2020-01-05"
        )
        q.append((True, df))
    except Exception as e:
        q.append((False, str(e)))


def test_qlib_operator(expr="Rank(Div($high, $low), 10)", verbose=True, timeout=60):
    import qlib

    instruments = "csi300"
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")

    fields = [expr]
    names = ["test_expr"]
    data_loader_config = {"feature": (fields, names)}

    result_container = []

    thread = threading.Thread(
        target=_load_data_thread, args=(result_container, data_loader_config)
    )
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        error_msg = f"Timeout: execution exceeded {timeout}s"
        if verbose:
            print(error_msg)
        return False, error_msg

    if not result_container:
        error_msg = "No result returned from data loader."
        if verbose:
            print(error_msg)
        return False, error_msg

    success, result = result_container[0]
    if not success:
        if verbose:
            print(f"Exception during operator test: {result}")
        return False, str(result)

    df = result
    ndf = df["feature"]
    if ndf.empty:
        error_msg = "Loaded DataFrame is empty. Which means factor doesn't contribute to any output. Test failed."
        if verbose:
            print(error_msg)
        return False, error_msg

    nan_ratio = ndf.isna().mean().mean()
    if nan_ratio > 0.01:
        error_msg = f"High NaN ratio: {nan_ratio:.2%}. Test failed."
        if verbose:
            print(error_msg)
        return False, error_msg
    elif nan_ratio > 0.001:
        if verbose:
            print(
                f"Warning: NaN ratio is {nan_ratio:.2%}, which is higher than expected but acceptable."
            )
        return True, None
    else:
        if verbose:
            print("Test passed: Valid expression with non-negligible data.")
        return True, None


def main():
    test_expr = "Rank(Div($high, $low), 10)"
    success, msg = test_qlib_operator(test_expr)
    print(
        f"Test result for expression `{test_expr}`: {'Success' if success else 'Failure'}"
    )


if __name__ == "__main__":
    main()
