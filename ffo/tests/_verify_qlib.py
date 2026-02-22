"""Verify the monkey-patch + Qlib data loading."""
import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.WARNING)

# ---- Monkey-patch (same as in backend_app.py / utils.py) ----
from qlib.utils.paral import ParallelExt
from joblib import Parallel

def _patched_init(self, *args, **kwargs):
    kwargs.pop("maxtasksperchild", None)
    kwargs["n_jobs"] = 1  # force serial on Windows
    Parallel.__init__(self, *args, **kwargs)

ParallelExt.__init__ = _patched_init
# ---- End monkey-patch ----

if __name__ == "__main__":
    import qlib
    from qlib.constant import REG_CN
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    print("qlib init OK")

    from qlib.data.dataset.loader import QlibDataLoader

    cfg = {
        "feature": (["$close"], ["alpha_close"]),
        "label": (["Ref($close, -1)/$close - 1"], ["LABEL"]),
    }
    loader = QlibDataLoader(config=cfg)
    df = loader.load(instruments="csi300", start_time="2022-01-01", end_time="2022-01-10")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(df.head(3))
    print("SUCCESS")
