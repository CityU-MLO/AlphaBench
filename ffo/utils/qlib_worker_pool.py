"""
Persistent Qlib Worker Pool — N processes per region (CN, US).

Each worker process calls qlib.init() once and loops forever, preserving
qlib's in-memory data cache (H) across requests.  Multiple workers per
region pull from a shared job_queue for true parallel backtest execution.

A dispatcher thread in the main process reads the shared result_queue and
routes results back to callers via threading Events — no Queue pickling.

Architecture:
    Main Process
        ├── Dispatcher thread (reads result_queue → wakes callers)
        └── QlibWorkerPool (router)
                ├── CN Workers (N Processes)  ← shared job_queue / result_queue
                └── US Workers (N Processes)  ← shared job_queue / result_queue
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

        if job is None:
            logger.info("Worker [%s] shutting down", tag)
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
            logger.warning("Worker [%s] job %s failed: %s", tag, job_id, e)
            result_queue.put({"job_id": job_id, "ok": False, "result": None, "error": f"{type(e).__name__}: {e}\n{tb}"})


# ---------------------------------------------------------------------------
#  Region worker group — manages N child processes for one region
# ---------------------------------------------------------------------------

class _RegionWorkerGroup:
    """
    Manages N persistent worker processes for a single region.

    All N workers share a single job_queue (for automatic load balancing)
    and a single result_queue.  A dispatcher thread reads result_queue and
    routes results to callers via a pending-jobs dict keyed by job_id.
    Each caller waits on its own threading.Event — no cross-talk.
    """

    def __init__(self, data_path: str, region: str, n_workers: int = 1):
        self.data_path = str(Path(data_path).expanduser())
        self.region = region
        self.n_workers = max(1, n_workers)

        self.job_queue: Optional[Queue] = None
        self.result_queue: Optional[Queue] = None
        self._processes: List[Optional[Process]] = []
        self._lock = threading.Lock()

        # Pending jobs: job_id → {"event": Event, "msg": result_dict | None}
        self._pending: Dict[str, dict] = {}
        self._pending_lock = threading.Lock()

        self._dispatcher_thread: Optional[threading.Thread] = None
        self._shutdown_flag = threading.Event()

        self._start_all()

    # -- lifecycle -----------------------------------------------------------

    def _start_all(self):
        """Spawn all worker processes and the dispatcher thread."""
        self.job_queue = Queue()
        self.result_queue = Queue()
        self._processes = []
        self._shutdown_flag.clear()

        for i in range(self.n_workers):
            self._spawn_worker(i)

        self._dispatcher_thread = threading.Thread(
            target=self._dispatch_loop, daemon=True,
            name=f"dispatcher-{self.region}",
        )
        self._dispatcher_thread.start()

        logger.info(
            "RegionWorkerGroup [%s] started %d workers + dispatcher",
            self.region, self.n_workers,
        )

    def _spawn_worker(self, idx: int) -> Process:
        """Spawn a single worker at the given index."""
        p = Process(
            target=_worker_loop,
            args=(self.data_path, self.region, idx, self.job_queue, self.result_queue),
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
        """Restart any dead workers (must hold self._lock)."""
        for i, p in enumerate(self._processes):
            if p is None or not p.is_alive():
                logger.warning("Worker [%s#%d] is dead, restarting…", self.region, i)
                self._spawn_worker(i)

    # -- dispatcher thread ---------------------------------------------------

    def _dispatch_loop(self):
        """
        Runs in a daemon thread.  Reads results from the shared result_queue
        and wakes up the caller that is waiting for each job_id.
        """
        while not self._shutdown_flag.is_set():
            try:
                msg = self.result_queue.get(timeout=1.0)
            except (queue.Empty, OSError):
                continue

            job_id = msg.get("job_id")
            if job_id is None:
                continue

            with self._pending_lock:
                slot = self._pending.get(job_id)

            if slot is not None:
                slot["msg"] = msg
                slot["event"].set()
            else:
                logger.warning(
                    "Dispatcher [%s]: no pending caller for job %s (timed out?)",
                    self.region, job_id[:8],
                )

    # -- job submission ------------------------------------------------------

    def submit(self, job_type: str, kwargs: dict, timeout: float = 300) -> Any:
        """
        Submit a job and block until the result arrives.

        Thread-safe: multiple callers can submit concurrently.  Each gets
        its own Event so there is no cross-talk.
        """
        with self._lock:
            self._ensure_workers_alive()

        job_id = str(uuid.uuid4())
        event = threading.Event()

        slot = {"event": event, "msg": None}
        with self._pending_lock:
            self._pending[job_id] = slot

        try:
            self.job_queue.put({
                "job_id": job_id,
                "type": job_type,
                "kwargs": kwargs,
            })

            if not event.wait(timeout=timeout):
                logger.warning(
                    "Region [%s] timed out on job %s after %ds",
                    self.region, job_id[:8], timeout,
                )
                raise TimeoutError(f"Backtest timed out after {timeout}s")

            msg = slot["msg"]
            if msg["ok"]:
                return msg["result"]
            else:
                raise RuntimeError(msg["error"])
        finally:
            with self._pending_lock:
                self._pending.pop(job_id, None)

    # -- alive / health ------------------------------------------------------

    def is_alive(self) -> bool:
        return any(p and p.is_alive() for p in self._processes)

    def alive_count(self) -> int:
        return sum(1 for p in self._processes if p and p.is_alive())

    # -- shutdown ------------------------------------------------------------

    def shutdown(self):
        """Graceful shutdown: send sentinels, wait, then cleanup."""
        self._shutdown_flag.set()

        # Send one sentinel per worker
        if self.job_queue is not None:
            for _ in self._processes:
                try:
                    self.job_queue.put(None)
                except Exception:
                    pass

        for p in self._processes:
            if p is not None:
                p.join(timeout=10)
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=5)
        self._processes.clear()

        if self._dispatcher_thread is not None:
            self._dispatcher_thread.join(timeout=5)
            self._dispatcher_thread = None

        for q in (self.job_queue, self.result_queue):
            if q is not None:
                try:
                    q.close()
                except Exception:
                    pass
        self.job_queue = None
        self.result_queue = None

        # Wake any callers still blocked
        with self._pending_lock:
            for slot in self._pending.values():
                if slot["msg"] is None:
                    slot["msg"] = {"job_id": "?", "ok": False, "result": None, "error": "Worker pool shut down"}
                slot["event"].set()
            self._pending.clear()


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
