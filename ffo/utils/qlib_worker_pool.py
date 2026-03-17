"""
Persistent Qlib Worker Pool — N processes per region (CN, US).

Each worker process calls qlib.init() once and loops forever, preserving
qlib's in-memory data cache (H) across requests.  A router dispatches
backtest jobs to the correct worker based on region.  Multiple workers per
region enables true parallel backtest execution.

Architecture:
    Main Process
        └── QlibWorkerPool (router)
                ├── CN Workers (N Processes)  — qlib.init(cn_data)
                └── US Workers (N Processes)  — qlib.init(us_data)
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
import traceback
import uuid
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("QlibWorkerPool")

DEFAULT_WORKERS_PER_REGION = max(1, (os.cpu_count() or 4) // 2)


# ---------------------------------------------------------------------------
#  Worker loop — runs inside child process
# ---------------------------------------------------------------------------

def _worker_loop(
    data_path: str,
    region: str,
    worker_idx: int,
    job_queue: Queue,
    result_queue: Queue,
):
    """
    Persistent worker event loop.

    Initialises qlib once, then processes jobs from job_queue forever.
    Sends results (or errors) back via result_queue keyed by job_id.
    """
    # Heavy imports inside the worker process only
    import qlib
    from backtest.qlib.single_alpha_backtest import (
        backtest_by_scores,
        backtest_by_single_alpha,
    )

    tag = f"{region}#{worker_idx}"
    try:
        qlib.init(provider_uri=data_path, region=region)
        logger.info("Worker [%s] initialised (data_path=%s)", tag, data_path)
    except Exception:
        logger.exception("Worker [%s] failed to init qlib", tag)
        return

    while True:
        try:
            job = job_queue.get()
        except Exception:
            break

        # Sentinel: None → graceful shutdown
        if job is None:
            logger.info("Worker [%s] shutting down", tag)
            break

        job_id = job.get("job_id", "?")
        job_type = job.get("type", "")
        kwargs = job.get("kwargs", {})
        reply_queue: Optional[Queue] = job.get("reply_queue")

        out_queue = reply_queue if reply_queue is not None else result_queue

        try:
            if job_type == "backtest_by_scores":
                result = backtest_by_scores(**kwargs)
            elif job_type == "backtest_by_single_alpha":
                result = backtest_by_single_alpha(**kwargs)
            else:
                raise ValueError(f"Unknown job type: {job_type}")

            out_queue.put({"job_id": job_id, "ok": True, "result": result, "error": None})

        except Exception as e:
            tb = traceback.format_exc()
            logger.warning("Worker [%s] job %s failed: %s", tag, job_id, e)
            out_queue.put({"job_id": job_id, "ok": False, "result": None, "error": f"{type(e).__name__}: {e}\n{tb}"})


# ---------------------------------------------------------------------------
#  Region worker group — manages N child processes for one region
# ---------------------------------------------------------------------------

class _RegionWorkerGroup:
    """
    Manages N persistent worker processes for a single region.

    All workers share a single job_queue so jobs are distributed to whichever
    worker is free first.  Each submitted job gets its own reply_queue so
    callers never steal each other's results.
    """

    def __init__(self, data_path: str, region: str, n_workers: int = 1):
        self.data_path = str(Path(data_path).expanduser())
        self.region = region
        self.n_workers = max(1, n_workers)

        self.job_queue: Optional[Queue] = None
        self._processes: List[Optional[Process]] = []
        self._lock = threading.Lock()

        self._start_all()

    # -- lifecycle -----------------------------------------------------------

    def _start_all(self):
        """Spawn all worker processes."""
        self.job_queue = Queue()
        self._processes = []
        for i in range(self.n_workers):
            self._spawn_worker(i)
        logger.info(
            "RegionWorkerGroup [%s] started %d workers", self.region, self.n_workers,
        )

    def _spawn_worker(self, idx: int) -> Process:
        """Spawn a single worker at the given index."""
        # Each worker shares the job_queue; reply_queue is per-job (sent in the job dict)
        # We still create a dummy result_queue for the worker_loop signature,
        # but it won't be used because every job carries its own reply_queue.
        dummy_result_queue = Queue()
        p = Process(
            target=_worker_loop,
            args=(self.data_path, self.region, idx, self.job_queue, dummy_result_queue),
            daemon=True,
            name=f"qlib-worker-{self.region}-{idx}",
        )
        p.start()
        logger.info("Spawned worker [%s#%d] pid=%d", self.region, idx, p.pid)
        if idx < len(self._processes):
            self._processes[idx] = p
        else:
            self._processes.append(p)
        return p

    def _ensure_workers_alive(self):
        """Restart any dead workers."""
        for i, p in enumerate(self._processes):
            if p is None or not p.is_alive():
                logger.warning("Worker [%s#%d] is dead, restarting…", self.region, i)
                self._spawn_worker(i)

    # -- job submission ------------------------------------------------------

    def submit(self, job_type: str, kwargs: dict, timeout: float = 300) -> Any:
        """
        Submit a job and wait for the result.

        Each call gets a dedicated reply_queue, so concurrent callers are
        completely isolated — no cross-talk or lost results.
        """
        with self._lock:
            self._ensure_workers_alive()

        job_id = str(uuid.uuid4())
        reply_queue: Queue = Queue()

        self.job_queue.put({
            "job_id": job_id,
            "type": job_type,
            "kwargs": kwargs,
            "reply_queue": reply_queue,
        })

        start_t = time.monotonic()
        while True:
            elapsed = time.monotonic() - start_t
            remaining = timeout - elapsed
            if remaining <= 0:
                logger.warning(
                    "Region [%s] timed out on job %s after %ds",
                    self.region, job_id[:8], timeout,
                )
                raise TimeoutError(f"Backtest timed out after {timeout}s")

            try:
                msg = reply_queue.get(timeout=min(remaining, 5.0))
            except queue.Empty:
                # Check that at least one worker is alive
                alive = any(p and p.is_alive() for p in self._processes)
                if not alive:
                    with self._lock:
                        self._ensure_workers_alive()
                    # Re-submit the job since all workers died
                    self.job_queue.put({
                        "job_id": job_id,
                        "type": job_type,
                        "kwargs": kwargs,
                        "reply_queue": reply_queue,
                    })
                continue

            if msg.get("job_id") == job_id:
                if msg["ok"]:
                    return msg["result"]
                else:
                    raise RuntimeError(msg["error"])
            # Shouldn't happen with per-job queues, but be safe
            logger.warning(
                "Got unexpected result for job %s on queue for %s",
                msg.get("job_id", "?")[:8], job_id[:8],
            )

    # -- alive / health ------------------------------------------------------

    def is_alive(self) -> bool:
        return any(p and p.is_alive() for p in self._processes)

    def alive_count(self) -> int:
        return sum(1 for p in self._processes if p and p.is_alive())

    # -- shutdown ------------------------------------------------------------

    def shutdown(self):
        """Graceful shutdown: send sentinels, wait, then cleanup."""
        if self.job_queue is not None:
            for _ in self._processes:
                try:
                    self.job_queue.put(None)  # sentinel per worker
                except Exception:
                    pass
        for p in self._processes:
            if p is not None:
                p.join(timeout=10)
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=5)
        self._processes.clear()
        if self.job_queue is not None:
            try:
                self.job_queue.close()
            except Exception:
                pass
            self.job_queue = None


# ---------------------------------------------------------------------------
#  Pool — public API
# ---------------------------------------------------------------------------

class QlibWorkerPool:
    """
    Manages persistent qlib worker processes, N per region.

    Usage:
        pool = QlibWorkerPool({
            "cn": {"data_path": "~/.qlib/qlib_data/cn_data", "region": "cn"},
            "us": {"data_path": "~/.qlib/qlib_data/us_data", "region": "us"},
        }, workers_per_region=4)

        result = pool.submit_backtest("cn", "backtest_by_scores", {
            "factor_scores": scores_df,
            "topk": 50, "n_drop": 5,
            "start_time": "2023-01-01", "end_time": "2024-01-01",
            "data_path": "~/.qlib/qlib_data/cn_data",
            "region": "cn", "BENCH": "SH000300",
        })
    """

    def __init__(
        self,
        region_configs: Dict[str, Dict[str, str]],
        workers_per_region: int = DEFAULT_WORKERS_PER_REGION,
    ):
        self._workers: Dict[str, _RegionWorkerGroup] = {}
        for region, cfg in region_configs.items():
            self._workers[region] = _RegionWorkerGroup(
                cfg["data_path"], cfg["region"], n_workers=workers_per_region,
            )
        logger.info(
            "QlibWorkerPool started — regions: %s, workers_per_region: %d",
            list(self._workers.keys()), workers_per_region,
        )

    def submit_backtest(
        self,
        region: str,
        job_type: str,
        kwargs: dict,
        timeout: float = 300,
    ) -> Tuple:
        """
        Submit a backtest job to an available worker for the given region.

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
            logger.info("Shutting down workers [%s]", region)
            w.shutdown()
        self._workers.clear()
