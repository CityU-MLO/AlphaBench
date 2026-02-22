"""
Execution engine for efficient multi-task processing.

Provides optimized executors for:
- CPU-intensive tasks (factor evaluation, IC computation) using ProcessPoolExecutor
- I/O-bound tasks (backtesting, API calls) using ThreadPoolExecutor
- Hybrid tasks with intelligent work distribution
- Resource pooling to avoid repeated initialization overhead

Usage in routes:
    >>> from utils.execution_engine import get_evaluation_executor, get_backtest_executor
    >>>
    >>> # CPU-intensive IC evaluation
    >>> executor = get_evaluation_executor()
    >>> futures = [executor.submit(evaluate_ic, expr) for expr in expressions]
    >>> results = [f.result() for f in as_completed(futures)]
    >>>
    >>> # I/O-bound backtesting
    >>> executor = get_backtest_executor()
    >>> futures = [executor.submit(run_backtest, expr) for expr in expressions]
    >>> results = [f.result() for f in as_completed(futures)]
"""

import os
import logging
import multiprocessing as mp
from concurrent.futures import (
    ThreadPoolExecutor,
    ProcessPoolExecutor,
    as_completed,
    Future,
)
from typing import List, Callable, Any, Optional, Dict, Tuple
from functools import lru_cache
import time

logger = logging.getLogger(__name__)

# Configuration
CPU_COUNT = os.cpu_count() or 4
MAX_PROCESS_WORKERS = max(1, CPU_COUNT - 1)  # Leave one CPU free
MAX_THREAD_WORKERS = max(8, CPU_COUNT * 2)  # More threads for I/O

# Global executor pools (lazy initialization)
_process_executor: Optional[ProcessPoolExecutor] = None
_thread_executor: Optional[ThreadPoolExecutor] = None
_backtest_executor: Optional[ThreadPoolExecutor] = None


def get_evaluation_executor(max_workers: Optional[int] = None) -> ProcessPoolExecutor:
    """
    Get the global process pool executor for CPU-intensive factor evaluation.

    Process pools are better for:
    - IC computation (CPU-bound)
    - Numerical calculations
    - Tasks with GIL contention

    Args:
        max_workers: Maximum number of workers (default: CPU_COUNT - 1)

    Returns:
        ProcessPoolExecutor instance

    Note:
        The executor is cached globally to avoid repeated process spawning overhead.
    """
    global _process_executor

    if _process_executor is None:
        workers = max_workers or MAX_PROCESS_WORKERS
        logger.info(f"Initializing process pool with {workers} workers")
        _process_executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context('fork') if hasattr(os, 'fork') else None,
        )

    return _process_executor


def get_thread_executor(max_workers: Optional[int] = None) -> ThreadPoolExecutor:
    """
    Get the global thread pool executor for general I/O-bound tasks.

    Thread pools are better for:
    - API calls
    - Database queries
    - File I/O
    - Tasks with waiting/blocking

    Args:
        max_workers: Maximum number of workers (default: CPU_COUNT * 2)

    Returns:
        ThreadPoolExecutor instance
    """
    global _thread_executor

    if _thread_executor is None:
        workers = max_workers or MAX_THREAD_WORKERS
        logger.info(f"Initializing thread pool with {workers} workers")
        _thread_executor = ThreadPoolExecutor(max_workers=workers)

    return _thread_executor


def get_backtest_executor(max_workers: Optional[int] = None) -> ThreadPoolExecutor:
    """
    Get a dedicated thread pool executor for backtesting tasks.

    Backtesting is I/O-bound (data loading from Qlib) so threads are preferred.
    We use a separate pool to avoid contention with other I/O tasks.

    Args:
        max_workers: Maximum number of workers (default: 8)

    Returns:
        ThreadPoolExecutor instance
    """
    global _backtest_executor

    if _backtest_executor is None:
        workers = max_workers or min(8, MAX_THREAD_WORKERS)
        logger.info(f"Initializing backtest pool with {workers} workers")
        _backtest_executor = ThreadPoolExecutor(max_workers=workers)

    return _backtest_executor


def cleanup_executors():
    """
    Cleanup all executor pools.

    Should be called on application shutdown.
    """
    global _process_executor, _thread_executor, _backtest_executor

    for name, executor in [
        ("process", _process_executor),
        ("thread", _thread_executor),
        ("backtest", _backtest_executor),
    ]:
        if executor:
            logger.info(f"Shutting down {name} executor")
            executor.shutdown(wait=True)

    _process_executor = None
    _thread_executor = None
    _backtest_executor = None


class BatchExecutor:
    """
    Smart batch executor that automatically distributes work across
    process and thread pools based on task characteristics.

    Example:
        >>> executor = BatchExecutor()
        >>> results = executor.execute_batch(
        ...     tasks=[(evaluate_ic, expr) for expr in expressions],
        ...     task_type='cpu',  # or 'io'
        ...     max_workers=8
        ... )
    """

    def __init__(self):
        self.stats = {
            "total_tasks": 0,
            "cpu_tasks": 0,
            "io_tasks": 0,
            "failed_tasks": 0,
        }

    def execute_batch(
        self,
        tasks: List[Tuple[Callable, tuple]],
        task_type: str = 'auto',
        max_workers: Optional[int] = None,
        timeout: Optional[float] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Any]:
        """
        Execute a batch of tasks with optimal executor selection.

        Args:
            tasks: List of (function, args) tuples
            task_type: 'cpu', 'io', or 'auto' (default: auto)
            max_workers: Maximum number of workers
            timeout: Timeout per task in seconds
            progress_callback: Optional callback(completed, total)

        Returns:
            List of results (same order as tasks)

        Example:
            >>> tasks = [(compute_ic, (expr, data)) for expr in expressions]
            >>> results = executor.execute_batch(tasks, task_type='cpu')
        """
        if not tasks:
            return []

        self.stats["total_tasks"] += len(tasks)

        # Select executor based on task type
        if task_type == 'cpu':
            executor = get_evaluation_executor(max_workers)
            self.stats["cpu_tasks"] += len(tasks)
        elif task_type == 'io':
            executor = get_thread_executor(max_workers)
            self.stats["io_tasks"] += len(tasks)
        else:
            # Auto-detect: use threads by default (safer)
            executor = get_thread_executor(max_workers)
            self.stats["io_tasks"] += len(tasks)

        # Submit all tasks
        futures = {}
        for i, (func, args) in enumerate(tasks):
            future = executor.submit(func, *args)
            futures[future] = i

        # Collect results in order
        results = [None] * len(tasks)
        completed = 0

        for future in as_completed(futures, timeout=timeout):
            task_idx = futures[future]

            try:
                result = future.result(timeout=timeout)
                results[task_idx] = result
            except Exception as e:
                logger.error(f"Task {task_idx} failed: {e}")
                results[task_idx] = {
                    "success": False,
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
                self.stats["failed_tasks"] += 1

            completed += 1
            if progress_callback:
                progress_callback(completed, len(tasks))

        return results

    def get_stats(self) -> Dict[str, int]:
        """Get execution statistics."""
        return self.stats.copy()


class ChunkedExecutor:
    """
    Executor that processes large batches in chunks to avoid memory issues
    and provide incremental results.

    Example:
        >>> executor = ChunkedExecutor(chunk_size=50)
        >>> for chunk_results in executor.execute_chunked(tasks):
        ...     process_partial_results(chunk_results)
    """

    def __init__(self, chunk_size: int = 50):
        self.chunk_size = chunk_size
        self.batch_executor = BatchExecutor()

    def execute_chunked(
        self,
        tasks: List[Tuple[Callable, tuple]],
        task_type: str = 'auto',
        max_workers: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ):
        """
        Execute tasks in chunks, yielding results incrementally.

        Args:
            tasks: List of (function, args) tuples
            task_type: 'cpu' or 'io'
            max_workers: Maximum number of workers
            progress_callback: Optional callback(completed, total)

        Yields:
            List of results for each chunk

        Example:
            >>> tasks = [(evaluate, (expr,)) for expr in expressions]
            >>> for chunk_results in executor.execute_chunked(tasks):
            ...     save_results(chunk_results)
        """
        total_tasks = len(tasks)
        completed = 0

        for i in range(0, total_tasks, self.chunk_size):
            chunk = tasks[i:i + self.chunk_size]

            # Progress callback for chunk
            def chunk_progress(done, total):
                nonlocal completed
                completed = i + done
                if progress_callback:
                    progress_callback(completed, total_tasks)

            # Execute chunk
            chunk_results = self.batch_executor.execute_batch(
                chunk,
                task_type=task_type,
                max_workers=max_workers,
                progress_callback=chunk_progress,
            )

            yield chunk_results


# ==================== Convenience Functions ====================

def parallel_map(
    func: Callable,
    items: List[Any],
    task_type: str = 'auto',
    max_workers: Optional[int] = None,
    timeout: Optional[float] = None,
    progress: bool = False,
) -> List[Any]:
    """
    Parallel map function - like map() but uses process/thread pools.

    Args:
        func: Function to apply to each item
        items: List of items to process
        task_type: 'cpu', 'io', or 'auto'
        max_workers: Maximum number of workers
        timeout: Timeout per task
        progress: Show progress (prints to stdout)

    Returns:
        List of results

    Example:
        >>> def square(x):
        ...     return x ** 2
        >>> results = parallel_map(square, [1, 2, 3, 4], task_type='cpu')
        >>> print(results)  # [1, 4, 9, 16]
    """
    tasks = [(func, (item,)) for item in items]

    progress_callback = None
    if progress:
        def progress_callback(done, total):
            print(f"\rProgress: {done}/{total} ({100*done//total}%)", end="")
            if done == total:
                print()  # New line when done

    executor = BatchExecutor()
    return executor.execute_batch(
        tasks,
        task_type=task_type,
        max_workers=max_workers,
        timeout=timeout,
        progress_callback=progress_callback,
    )


def parallel_starmap(
    func: Callable,
    items: List[tuple],
    task_type: str = 'auto',
    max_workers: Optional[int] = None,
    timeout: Optional[float] = None,
    progress: bool = False,
) -> List[Any]:
    """
    Parallel starmap - like itertools.starmap() but parallel.

    Args:
        func: Function to apply
        items: List of argument tuples
        task_type: 'cpu', 'io', or 'auto'
        max_workers: Maximum number of workers
        timeout: Timeout per task
        progress: Show progress

    Returns:
        List of results

    Example:
        >>> def add(a, b):
        ...     return a + b
        >>> results = parallel_starmap(add, [(1, 2), (3, 4)], task_type='cpu')
        >>> print(results)  # [3, 7]
    """
    tasks = [(func, args) for args in items]

    progress_callback = None
    if progress:
        def progress_callback(done, total):
            print(f"\rProgress: {done}/{total} ({100*done//total}%)", end="")
            if done == total:
                print()

    executor = BatchExecutor()
    return executor.execute_batch(
        tasks,
        task_type=task_type,
        max_workers=max_workers,
        timeout=timeout,
        progress_callback=progress_callback,
    )
