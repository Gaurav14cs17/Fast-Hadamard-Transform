"""Shared utilities for all use-case benchmarks."""

import time
import math
import torch
import torch.nn.functional as F

try:
    from tabulate import tabulate as _tabulate
    def print_table(rows, headers):
        print(_tabulate(rows, headers=headers, tablefmt="grid"))
except ImportError:
    def print_table(rows, headers):
        widths = [max(len(str(h)), max(len(str(r[i])) for r in rows))
                  for i, h in enumerate(headers)]
        fmt = " | ".join(f"{{:>{w}}}" for w in widths)
        print(fmt.format(*headers))
        print("-" * (sum(widths) + 3 * (len(widths) - 1)))
        for r in rows:
            print(fmt.format(*[str(v) for v in r]))


def fast_hadamard_vectorized(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Vectorized butterfly FHT. O(n log n), matches scipy convention."""
    n = x.shape[-1]
    assert n & (n - 1) == 0, f"Dimension must be power of 2, got {n}"
    x = x.clone().float()
    log_n = int(math.log2(n))
    for i in range(log_n):
        half = 2 ** i
        full = 2 * half
        x = x.view(*x.shape[:-1], n // full, 2, half)
        a = x[..., 0, :].clone()
        b = x[..., 1, :].clone()
        x[..., 0, :] = a + b
        x[..., 1, :] = a - b
        x = x.view(*x.shape[:-3], n)
    return x * scale


def format_bytes(b):
    if b < 1024:
        return f"{b} B"
    elif b < 1024**2:
        return f"{b/1024:.1f} KB"
    elif b < 1024**3:
        return f"{b/1024**2:.1f} MB"
    else:
        return f"{b/1024**3:.1f} GB"


def next_power_of_2(n):
    return 1 << (n - 1).bit_length()


def time_fn(fn, *args, warmup=3, repeats=10, **kwargs):
    """Time a function, return mean time in ms."""
    for _ in range(warmup):
        fn(*args, **kwargs)
    t0 = time.perf_counter()
    for _ in range(repeats):
        result = fn(*args, **kwargs)
    elapsed = (time.perf_counter() - t0) / repeats * 1000
    return elapsed, result


def section_header(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print()
