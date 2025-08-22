from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime
import traceback
import logging

import os
import re
import sys
import qlib
import json
import traceback
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from urllib.parse import unquote
from flask import jsonify, request
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # optional, for progress bar
from qlib.data.dataset.loader import QlibDataLoader
from backtest.factor_metrics import (
    FL_Ic,
    FL_RankIc,
    FL_Ir,
    FL_Icir,
    FL_RankIcir,
    FL_QuantileReturn,
    FL_Turnover,
)
from tqdm import tqdm
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import traceback

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import evaluation functions
from backtest.qlib.dataloader import compute_factor_data
from backtest.factor_metrics import get_performance
import threading


def qlib_load_data_thread(q, config):
    try:
        data_loader = QlibDataLoader(config=config)
        df = data_loader.load(
            instruments="csi300", start_time="2020-01-01", end_time="2020-02-01"
        )
        q.append((True, df))
    except Exception as e:
        q.append((False, str(e)))


def compute_ic_for_col(data, col, metric_fn, returns):
    """Safely compute IC/RankIC per date for one factor column."""
    try:
        ic_per_date = (
            data["feature"][col]
            .groupby("datetime")
            .apply(lambda s: metric_fn(s, returns.loc[s.index]))
        )
        return col, ic_per_date
    except Exception as e:
        print(f"Error in column {col}: {e}")
        traceback.print_exc()
        return col, None


def summarize_ic_tables(ic_table: pd.DataFrame, rankic_table: pd.DataFrame):

    summary = []
    ic_table = ic_table.fillna(0)
    rankic_table = rankic_table.fillna(0)

    for factor in ic_table.columns:
        ic_values = ic_table[factor]
        rankic_values = rankic_table[factor]

        ic_mean = ic_values.mean()
        ic_std = ic_values.std()
        rankic_mean = rankic_values.mean()
        rankic_std = rankic_values.std()

        try:
            summary.append(
                {
                    "name": factor,
                    "success": True,
                    "timestamp": datetime.now().isoformat(),
                    "metrics": {
                        "ic": float(ic_mean),
                        "icir": float(ic_mean / ic_std) if ic_std > 0 else float("nan"),
                        "rank_ic": float(rankic_mean),
                        "rank_icir": float(rankic_mean / rankic_std)
                        if rankic_std > 0
                        else float("nan"),
                    },
                }
            )
        except Exception as e:
            summary.append(
                {
                    "name": factor,
                    "success": False,
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e),
                    "metrics": {"ic": 0, "icir": 0, "rank_ic": 0, "rank_icir": 0},
                }
            )

    return summary
