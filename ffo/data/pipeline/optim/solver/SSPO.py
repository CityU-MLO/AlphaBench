import numpy as np


def simplex_projection_selfnorm2(v, b=1):
    """Project vector v onto simplex sum(v)=b, v>=0."""
    v = np.array(v, dtype=np.float64)
    while np.max(np.abs(v)) > 1e6:
        v /= 10.0
    u = np.sort(v)[::-1]
    sv = np.cumsum(u)
    rho = np.where(u > (sv - b) / (np.arange(1, len(u) + 1)))[0][-1]
    theta = (sv[rho] - b) / (rho + 1)
    w = np.maximum(v - theta, 0)
    return w


def wthresh(x, thresh):
    """Soft thresholding."""
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0)


def SSPO_single_step(data_close, data, b_t_hat, opts):
    """
    Single-step SSPO solver for one time window.

    Args:
        data_close (np.ndarray): cumulative matrix (win_size x N)
        data (np.ndarray): price relatives / IC matrix (win_size x N)
        b_t_hat (np.ndarray): previous weight vector (N x 1)
        opts (dict): parameters
            - lambda
            - gamma
            - eta
            - zeta
            - ABSTOL
            - max_iter

    Returns:
        b_tplus1_hat (np.ndarray): updated weights
        prim_res (list): residuals
        n_iter (int): number of iterations
    """

    # ---- parameters ----
    max_iter = opts.get("max_iter", 1000)
    ABSTOL = opts.get("ABSTOL", 1e-4)
    zeta = opts.get("zeta", 500)
    lambda_ = opts.get("lambda", 0.5)
    gamma = opts.get("gamma", 0.01)
    eta = opts.get("eta", 0.005)

    tao = lambda_ / gamma
    nstk = data.shape[1]

    # ---- feature construction ----
    Rpredict = np.max(data_close, axis=0)
    x_tplus1 = Rpredict / data_close[-1, :]
    x_tplus1 = 1.1 * np.log(x_tplus1 + 1e-8) + 1
    x = -x_tplus1.reshape(-1, 1)

    # ---- initialize variables ----
    g = b_t_hat.copy()
    b = b_t_hat.copy()
    rho = 0.0
    I = np.eye(nstk)
    YI = np.ones((nstk, nstk))
    yi = np.ones((nstk, 1))
    prim_res = []

    # ---- main loop ----
    for iter_ in range(1, max_iter + 1):
        b = np.linalg.solve(tao * I + eta * YI, tao * g + (eta - rho) * yi - x)
        g = wthresh(b, gamma)
        prim_res_tmp = float(yi.T @ b - 1)
        rho += eta * prim_res_tmp
        prim_res.append(prim_res_tmp)
        if abs(prim_res_tmp) < ABSTOL:
            break

    # ---- normalization ----
    b_tplus1_hat = zeta * b.flatten()
    b_tplus1_hat = simplex_projection_selfnorm2(b_tplus1_hat, 1)

    return b_tplus1_hat, prim_res, iter_


if __name__ == "__main__":
    win_size = 63
    N = 32

    data = np.random.uniform(0.95, 1.05, size=(win_size, N))
    data_close = np.cumprod(data, axis=0)
    b_t_hat = np.ones((N, 1)) / N

    opts = {
        "max_iter": 10,
        "ABSTOL": 1e-4,
        "zeta": 500,
        "lambda": 0.001,
        "gamma": 0.01,
        "eta": 0.005,
    }

    b_next, prim_res, n_iter = SSPO_single_step(data_close, data, b_t_hat, opts)

    print("Next-step weights:", np.round(b_next, 4))
    print("Iterations:", n_iter)
    print("Final constraint residual:", prim_res[-1])
