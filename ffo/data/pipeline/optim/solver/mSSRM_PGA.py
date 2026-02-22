import numpy as np
import pandas as pd


def mSSRM_PGA_single_step(
    factor_window: np.ndarray, m: int, eps=1e-3, iternum=int(1e4), tol=1e-5
):
    """
    Single-step sparse optimizer (mSSRM-PGA) for one window of factor data.

    Parameters
    ----------
    factor_window : np.ndarray
        (T x N) matrix of factor values or ICs in the current window.
    m : int
        Number of non-zero factors to keep.
    eps : float
        Small constant for numerical regularization.
    iternum : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    w : np.ndarray
        Optimal sparse weight vector (N x 1) normalized to sum=1.
    n_iter : int
        Number of iterations performed.
    rel_error : float
        Final relative error.
    """

    T, N = factor_window.shape
    eI = eps * np.eye(N)

    # mean vector (expected IC or expected return)
    vecmu = np.mean(factor_window, axis=0).reshape(-1, 1)

    # centered matrix Q (same as original)
    Q = (1 / np.sqrt(T - 1)) * (factor_window - np.mean(factor_window, axis=0))
    QeI = Q.T @ Q + eI

    alpha = 0.999 / np.linalg.norm(QeI, 2)

    w = vecmu.copy()
    rel_error = 1e9
    k = 0

    while k < iternum and rel_error > tol:
        w_prev = w.copy()
        w_pre = w - alpha * (QeI @ w - vecmu)
        w_pre[w_pre < 0] = 0

        # keep top-m entries
        idx = np.argsort(w_pre.flatten())[::-1]
        w = np.zeros((N, 1))
        w[idx[:m]] = w_pre[idx[:m]]

        rel_error = np.linalg.norm(w - w_prev) / (np.linalg.norm(w_prev) + 1e-12)
        k += 1

    # normalize
    w_sum = np.sum(w)
    if w_sum > 0:
        w /= w_sum

    return w, k, rel_error


if __name__ == "__main__":
    # Suppose you have an IC table (dates × factors)
    dates = pd.date_range("2022-01-01", periods=60)
    factors = [f"factor_{i}" for i in range(32)]
    ic_df = pd.DataFrame(np.random.randn(60, 32) * 0.05, index=dates, columns=factors)

    # pick the last 30 days as one window
    window_data = ic_df.iloc[-30:].values

    # single-step sparse weight optimization
    w, n_iter, err = mSSRM_PGA_single_step(window_data, m=10)

    print("Optimal weights:", np.round(w.flatten(), 4))
    print("Nonzero count:", np.sum(w > 0))
    print("Iterations:", n_iter, "RelError:", err)
