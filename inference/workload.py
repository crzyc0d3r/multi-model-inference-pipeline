"""Bounded workers and honest operation-level latency/failure accounting."""

from __future__ import annotations

import math
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Callable


def run_workload(operation: Callable[[int], dict], count: int, concurrency: int) -> dict:
    if count < 1 or concurrency < 1:
        raise ValueError("count and concurrency must be positive")
    lock = threading.Lock()
    next_index = 0
    active = maximum_active = successes = 0
    latencies: list[float] = []
    errors: Counter[str] = Counter()
    samples: dict[int, dict] = {}
    start = time.perf_counter()

    def worker() -> None:
        nonlocal next_index, active, maximum_active, successes
        while True:
            with lock:
                if next_index >= count:
                    return
                index = next_index
                next_index += 1
                active += 1
                maximum_active = max(maximum_active, active)
            began = time.perf_counter()
            try:
                result = operation(index)
                with lock:
                    successes += 1
                    if index < 3:
                        samples[index] = result
            except Exception as exc:
                # Error messages from the client omit URLs, prompts, and keys.
                with lock:
                    errors[type(exc).__name__ + ": " + str(exc)] += 1
            finally:
                elapsed_ms = (time.perf_counter() - began) * 1000
                with lock:
                    active -= 1
                    latencies.append(elapsed_ms)

    # Only concurrency futures exist, even for a million operations.
    with ThreadPoolExecutor(max_workers=min(count, concurrency)) as executor:
        futures = [executor.submit(worker) for _ in range(min(count, concurrency))]
        for future in futures:
            future.result()
    elapsed = time.perf_counter() - start
    latencies.sort()

    def percentile(fraction: float) -> float:
        return round(latencies[max(0, math.ceil(len(latencies) * fraction) - 1)], 3)

    return {"operations": count, "succeeded": successes, "failed": count - successes,
            "concurrency_limit": concurrency, "max_in_flight": maximum_active,
            "elapsed_seconds": round(elapsed, 4),
            "completed_operations_per_second": round(count / elapsed, 2),
            "latency_ms": {"mean": round(sum(latencies) / len(latencies), 3),
                           "p50": percentile(.5), "p95": percentile(.95), "max": round(latencies[-1], 3)},
            "errors": dict(errors), "samples": [samples[k] for k in sorted(samples)]}
