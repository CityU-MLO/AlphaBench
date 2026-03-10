"""
Persistent Qlib Worker Pool — one process per region (CN, US).

Each worker process calls qlib.init() once and loops forever, preserving
qlib's in-memory data cache (H) across requests.  A router dispatches
backtest jobs to the correct worker based on region.

Architecture:
    Main Process
        └── QlibWorkerPool (router)
                ├── CN Worker (Process)  — qlib.init(cn_data)
                └── US Worker (Process)  — qlib.init(us_data)
"""

from __future__ import annotations

import logging
import os
import queue
import traceback
import uuid
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("QlibWorkerPool")


# ---------------------------------------------------------------------------
#  Worker loop — runs inside child process
# ---------------------------------------------------------------------------

def _worker_loop(data_path: str, region: str, job_queue: Queue, result_queue: Queue):
    """
    Persistent worker event loop.

    Initialises qlib once, then processes jobs from job_queue forever.
    Sends results (or errors) back via result_queue.
    """
    # Heavy imports inside the worker process only
    import qlib
    from backtest.qlib.single_alpha_backtest import (
        backtest_by_scores,
        backtest_by_single_alpha,
    )

    try:
        qlib.init(provider_uri=data_path, region=region)
        logger.info("Worker [%s] initialised (data_path=%s)", region, data_path)
    except Exception:
        logger.exception("Worker [%s] failed to init qlib", region)
        return

    while True:
        try:
            job = job_queue.get()
        except Exception:
            break

        # Sentinel: None → graceful shutdown
        if job is None:
            logger.info("Worker [%s] shutting down", region)
            break

        job_id = job.get("job_id", "?")
        job_type = job.get("type", "")
        kwargs = job.get("kwargs", {})

        try:
            if job_type == "backtest_by_scores":
                result = backtest_by_scores(**kwargs)
            elif job_type == "backtest_by_single_alpha":
                result = backtest_by_single_alpha(**kwargs)
            else:
                raise ValueError(f"Unknown job type: {job_type}")

            result_queue.put({"job_id": job_id, "ok": True, "result": result, "error": None})

        except Exception as e:
            tb = traceback.format_exc()
            logger.warning("Worker [%s] job %s failed: %s", region, job_id, e)
            result_queue.put({"job_id": job_id, "ok": False, "result": None, "error": f"{type(e).__name__}: {e}\n{tb}"})


# ---------------------------------------------------------------------------
#  Worker handle — manages one child process
# ---------------------------------------------------------------------------

class _WorkerHandle:
    """Wraps a single persistent worker process."""

    def __init__(self, data_path: str, region: str):
        self.data_path = str(Path(data_path).expanduser())
        self.region = region
        self.job_queue: Optional[Queue] = None
        self.result_queue: Optional[Queue] = None
        self._process: Optional[Process] = None
        self._start()

    def _start(self):
        """Spawn (or re-spawn) the worker process."""
        self.job_queue = Queue()
        self.result_queue = Queue()
        self._process = Process(
            target=_worker_loop,
            args=(self.data_path, self.region, self.job_queue, self.result_queue),
            daemon=True,
            name=f"qlib-worker-{self.region}",
        )
        self._process.start()
        logger.info("Spawned worker [%s] pid=%d", self.region, self._process.pid)

    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def ensure_alive(self):
        """Restart worker if it has died."""
        if not self.is_alive():
            logger.warning("Worker [%s] is dead, restarting…", self.region)
            self._cleanup()
            self._start()

    def submit(self, job_type: str, kwargs: dict, timeout: float = 300) -> Any:
        """
        Submit a job and wait for the result.

        Returns the job result (e.g. tuple of DataFrames).
        Raises TimeoutError or RuntimeError on failure.
        """
        self.ensure_alive()

        job_id = str(uuid.uuid4())
        self.job_queue.put({"job_id": job_id, "type": job_type, "kwargs": kwargs})

        # Wait for result — drain queue until we find our job_id
        # (in single-submitter-per-worker scenarios this is always the first)
        deadline_remaining = timeout
        import time
        start_t = time.monotonic()

        while True:
            elapsed = time.monotonic() - start_t
            remaining = timeout - elapsed
            if remaining <= 0:
                # Timeout — kill and restart worker
                logger.warning("Worker [%s] timed out on job %s, restarting", self.region, job_id[:8])
                self._kill_and_restart()
                raise TimeoutError(f"Backtest timed out after {timeout}s")

            try:
                msg = self.result_queue.get(timeout=min(remaining, 5.0))
            except queue.Empty:
                # Check if worker died while we wait
                if not self.is_alive():
                    self._cleanup()
                    self._start()
                    raise RuntimeError(f"Worker [{self.region}] died during job execution")
                continue

            if msg.get("job_id") == job_id:
                if msg["ok"]:
                    return msg["result"]
                else:
                    raise RuntimeError(msg["error"])
            else:
                # Not our job — shouldn't happen in single-submitter mode
                # but put it back defensively
                logger.warning("Got stale result for job %s, expected %s", msg.get("job_id", "?")[:8], job_id[:8])

    def _kill_and_restart(self):
        """Hard-kill the worker and spawn a fresh one."""
        if self._process and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=2)
        self._cleanup()
        self._start()

    def _cleanup(self):
        """Close queues and process handle."""
        for q in (self.job_queue, self.result_queue):
            if q is not None:
                try:
                    q.close()
                except Exception:
                    pass
        self._process = None
        self.job_queue = None
        self.result_queue = None

    def shutdown(self):
        """Graceful shutdown: send sentinel, wait, then cleanup."""
        if self.job_queue is not None and self.is_alive():
            try:
                self.job_queue.put(None)  # sentinel
            except Exception:
                pass
        if self._process is not None:
            self._process.join(timeout=10)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=5)
        self._cleanup()


# ---------------------------------------------------------------------------
#  Pool — public API
# ---------------------------------------------------------------------------

class QlibWorkerPool:
    """
    Manages persistent qlib worker processes, one per region.

    Usage:
        pool = QlibWorkerPool({
            "cn": {"data_path": "~/.qlib/qlib_data/cn_data", "region": "cn"},
            "us": {"data_path": "~/.qlib/qlib_data/us_data", "region": "us"},
        })

        result = pool.submit_backtest("cn", "backtest_by_scores", {
            "factor_scores": scores_df,
            "topk": 50, "n_drop": 5,
            "start_time": "2023-01-01", "end_time": "2024-01-01",
            "data_path": "~/.qlib/qlib_data/cn_data",
            "region": "cn", "BENCH": "SH000300",
        })
    """

    def __init__(self, region_configs: Dict[str, Dict[str, str]]):
        self._workers: Dict[str, _WorkerHandle] = {}
        for region, cfg in region_configs.items():
            self._workers[region] = _WorkerHandle(cfg["data_path"], cfg["region"])
        logger.info("QlibWorkerPool started with regions: %s", list(self._workers.keys()))

    def submit_backtest(
        self,
        region: str,
        job_type: str,
        kwargs: dict,
        timeout: float = 300,
    ) -> Tuple:
        """
        Submit a backtest job to the worker for the given region.

        Args:
            region: "cn" or "us"
            job_type: "backtest_by_scores" or "backtest_by_single_alpha"
            kwargs: arguments passed to the backtest function
            timeout: max seconds to wait

        Returns:
            (analysis_df, report_normal, positions_normal)
        """
        if region not in self._workers:
            raise ValueError(f"No worker for region '{region}'. Available: {list(self._workers.keys())}")
        return self._workers[region].submit(job_type, kwargs, timeout=timeout)

    def is_alive(self, region: str) -> bool:
        w = self._workers.get(region)
        return w.is_alive() if w else False

    def shutdown(self):
        """Gracefully stop all workers."""
        for region, w in self._workers.items():
            logger.info("Shutting down worker [%s]", region)
            w.shutdown()
        self._workers.clear()
