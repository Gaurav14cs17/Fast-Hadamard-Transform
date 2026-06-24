"""
USE CASE 1: Random Projection via Structured Random Hadamard Transform (SRHT)

Problem: Dimensionality reduction while preserving pairwise distances.
Classical approach: Multiply by a dense Gaussian random matrix R ∈ ℝ^{k×n}
SRHT approach: D (random signs) → H (Hadamard) → sample k coordinates

The SRHT achieves the same Johnson-Lindenstrauss guarantee with:
  - O(n log n) time instead of O(nk) for dense projection
  - O(n) memory instead of O(nk) for the projection matrix
"""

import time
import math
import torch
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from use_cases.common import (
    fast_hadamard_vectorized, format_bytes, print_table,
    time_fn, section_header, next_power_of_2
)


def dense_gaussian_projection(x, k):
    """Classical JL: multiply by dense Gaussian matrix."""
    n = x.shape[-1]
    R = torch.randn(k, n, device=x.device) / math.sqrt(k)
    return x @ R.T


def srht_projection(x, k):
    """Structured Random Hadamard Transform projection."""
    n = x.shape[-1]
    n_pad = next_power_of_2(n)

    D = torch.sign(torch.randn(n, device=x.device))
    D[D == 0] = 1.0
    x_signed = x * D.unsqueeze(0)

    if n != n_pad:
        x_signed = torch.nn.functional.pad(x_signed, (0, n_pad - n))

    x_hadamard = fast_hadamard_vectorized(x_signed, scale=1.0 / math.sqrt(n_pad))
    indices = torch.randperm(n_pad, device=x.device)[:k]
    return x_hadamard[:, indices] * math.sqrt(n_pad / k)


def sparse_rademacher_projection(x, k, sparsity=3):
    """Sparse JL: each entry of R is ±1/√s with prob 1/(2s), 0 otherwise."""
    n = x.shape[-1]
    s = sparsity
    R = torch.zeros(k, n, device=x.device)
    mask = torch.rand(k, n, device=x.device) < (1.0 / s)
    signs = torch.sign(torch.randn(k, n, device=x.device))
    R[mask] = signs[mask] * math.sqrt(s / k)
    return x @ R.T


def run():
    section_header("USE CASE 1: Random Projection (Johnson-Lindenstrauss)")

    print("  Goal: Reduce dimension from n → k while preserving distances.")
    print("  JL Lemma guarantees: (1-ε)||u-v|| ≤ ||f(u)-f(v)|| ≤ (1+ε)||u-v||")
    print()

    # --- Part A: Timing comparison ---
    print("  ─── A. Speed Comparison ───")
    print()

    configs = [
        (512, 64, 256),
        (1024, 128, 256),
        (2048, 256, 128),
        (4096, 512, 64),
        (8192, 1024, 32),
    ]

    rows = []
    for n, k, batch in configs:
        x = torch.randn(batch, n)

        t_dense, _ = time_fn(dense_gaussian_projection, x, k)
        t_srht, _ = time_fn(srht_projection, x, k)
        t_sparse, _ = time_fn(sparse_rademacher_projection, x, k)

        speedup = t_dense / t_srht
        rows.append([
            f"{n}→{k}",
            batch,
            f"{t_dense:.2f} ms",
            f"{t_sparse:.2f} ms",
            f"{t_srht:.2f} ms",
            f"{speedup:.1f}x",
        ])

    print_table(rows, ["n→k", "Batch", "Dense Gauss", "Sparse JL", "SRHT (Hadamard)", "SRHT speedup"])
    print()

    # --- Part B: Memory comparison ---
    print("  ─── B. Memory Comparison ───")
    print()
    rows = []
    for n, k, _ in configs:
        dense_mem = k * n * 4
        sparse_mem = k * n * 4 // 3  # ~1/3 nonzero
        srht_mem = n * 4  # just the sign vector
        rows.append([
            f"{n}→{k}",
            format_bytes(dense_mem),
            format_bytes(sparse_mem),
            format_bytes(srht_mem),
            f"{dense_mem // max(srht_mem, 1)}x less",
        ])

    print_table(rows, ["n→k", "Dense R matrix", "Sparse R matrix", "SRHT (sign vec)", "SRHT advantage"])
    print()

    # --- Part C: Distance preservation quality ---
    print("  ─── C. Distance Preservation Quality ───")
    print()
    print("  Measuring how well each method preserves pairwise distances.")
    print("  (Lower distortion std = better quality)")
    print()

    n, k, batch = 1024, 128, 200
    x = torch.randn(batch, n)

    num_pairs = min(500, batch * (batch - 1) // 2)
    idx_i = torch.randint(0, batch, (num_pairs,))
    idx_j = torch.randint(0, batch, (num_pairs,))
    mask = idx_i != idx_j
    idx_i, idx_j = idx_i[mask], idx_j[mask]

    orig_dists = torch.norm(x[idx_i] - x[idx_j], dim=1)

    methods = {
        "Dense Gaussian": dense_gaussian_projection,
        "Sparse Rademacher": sparse_rademacher_projection,
        "SRHT (Hadamard)": srht_projection,
    }

    rows = []
    for name, fn in methods.items():
        ratios_all = []
        for trial in range(5):
            if name == "Sparse Rademacher":
                proj = fn(x, k, sparsity=3)
            else:
                proj = fn(x, k)
            proj_dists = torch.norm(proj[idx_i] - proj[idx_j], dim=1)
            ratios = (proj_dists / orig_dists).numpy()
            ratios_all.extend(ratios.tolist())

        ratios_arr = np.array(ratios_all)
        rows.append([
            name,
            f"{ratios_arr.mean():.4f}",
            f"{ratios_arr.std():.4f}",
            f"{abs(ratios_arr.mean() - 1.0):.4f}",
            f"[{ratios_arr.min():.3f}, {ratios_arr.max():.3f}]",
        ])

    print_table(rows, ["Method", "Mean ratio", "Std (lower=better)", "Bias", "Range"])
    print()
    print("  Ideal: mean ratio ≈ 1.0, low std, tight range around 1.0")
    print()


if __name__ == "__main__":
    run()
