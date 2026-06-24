"""
USE CASE 4: Hadamard as a Cheap Linear Layer in Neural Networks

Problem: Dense linear layers (nn.Linear) are O(n²) in both compute and parameters.
For mixing/shuffling features between channels, a full learned matrix is overkill.

Solution: Replace dense layers with Hadamard transforms where "mixing" (not learning)
is the goal. This gives O(n log n) feature interaction with ZERO learned parameters.

Real examples:
  - FNet: replaces attention with Fourier/Hadamard mixing
  - Monarch Mixer: uses structured matrices including Hadamard
  - HyperMixer: Hadamard-based token mixing
  - Efficient MLPs: Hadamard + diagonal (learnable) ≈ full linear
"""

import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from use_cases.common import (
    fast_hadamard_vectorized, format_bytes, print_table,
    time_fn, section_header, next_power_of_2
)


class DenseLinearLayer(nn.Module):
    """Standard nn.Linear: O(n²) params and compute."""
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.linear(x)


class HadamardLayer(nn.Module):
    """Pure Hadamard mixing: 0 params, O(n log n) compute."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.scale = 1.0 / math.sqrt(dim)

    def forward(self, x):
        return fast_hadamard_vectorized(x, scale=self.scale)


class HadamardDiagonalLayer(nn.Module):
    """
    Hadamard + learnable diagonal: O(n) params, O(n log n) compute.
    Approximates a full dense layer: out = H @ diag(d) @ x
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.scale = 1.0 / math.sqrt(dim)
        self.diag = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x * self.diag
        return fast_hadamard_vectorized(x, scale=self.scale)


class HadamardSandwichLayer(nn.Module):
    """
    H @ diag(d1) @ H @ diag(d2) @ x — two Hadamard passes with diagonals.
    O(2n) params, O(2n log n) compute. Approximates rank-n dense very well.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.scale = 1.0 / math.sqrt(dim)
        self.d1 = nn.Parameter(torch.randn(dim) * 0.02)
        self.d2 = nn.Parameter(torch.randn(dim) * 0.02)

    def forward(self, x):
        x = x * self.d2
        x = fast_hadamard_vectorized(x, scale=self.scale)
        x = x * self.d1
        x = fast_hadamard_vectorized(x, scale=self.scale)
        return x


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def run():
    section_header("USE CASE 4: Hadamard as Neural Network Layer")

    print("  Comparing layer architectures for feature mixing:")
    print("    • Dense (nn.Linear): O(n²) params, O(n²) compute")
    print("    • Hadamard only: 0 params, O(n log n) compute")
    print("    • Hadamard + diagonal: O(n) params, O(n log n) compute")
    print("    • Hadamard sandwich (H·D·H·D): O(2n) params, O(2n log n) compute")
    print()

    # --- Part A: Parameter & Compute comparison ---
    print("  ─── A. Parameter Count & Theoretical FLOPs ───")
    print()

    dims = [128, 256, 512, 1024, 2048, 4096]
    rows = []
    for dim in dims:
        dense_params = dim * dim
        had_params = 0
        had_diag_params = dim
        sandwich_params = 2 * dim

        dense_flops = 2 * dim * dim  # matmul
        had_flops = dim * int(math.log2(dim))
        had_diag_flops = dim + dim * int(math.log2(dim))
        sandwich_flops = 2 * (dim + dim * int(math.log2(dim)))

        rows.append([
            dim,
            f"{dense_params:,} ({format_bytes(dense_params*4)})",
            f"{had_params}",
            f"{had_diag_params:,} ({format_bytes(had_diag_params*4)})",
            f"{sandwich_params:,} ({format_bytes(sandwich_params*4)})",
            f"{dense_params // max(sandwich_params, 1)}x",
        ])

    print_table(rows, ["Dim", "Dense params", "Hadamard", "H+Diag", "Sandwich", "Dense/Sandwich"])
    print()

    # --- Part B: Actual timing ---
    print("  ─── B. Forward Pass Timing ───")
    print()

    batch = 128
    rows = []
    for dim in [128, 256, 512, 1024, 2048]:
        x = torch.randn(batch, dim)

        models = {
            "Dense": DenseLinearLayer(dim),
            "Hadamard": HadamardLayer(dim),
            "H+Diag": HadamardDiagonalLayer(dim),
            "Sandwich": HadamardSandwichLayer(dim),
        }

        times = {}
        for name, model in models.items():
            model.eval()
            with torch.no_grad():
                t, _ = time_fn(model, x, warmup=5, repeats=20)
            times[name] = t

        rows.append([
            f"{batch}×{dim}",
            f"{times['Dense']:.3f} ms",
            f"{times['Hadamard']:.3f} ms",
            f"{times['H+Diag']:.3f} ms",
            f"{times['Sandwich']:.3f} ms",
            f"{times['Dense'] / times['Sandwich']:.1f}x",
        ])

    print_table(rows, ["Input", "Dense", "Hadamard", "H+Diag", "Sandwich", "Dense/Sandwich"])
    print()

    # --- Part C: Expressivity test (function approximation) ---
    print("  ─── C. Expressivity: Learning a Random Target Function ───")
    print()
    print("  Train each layer type to approximate a random dense linear mapping.")
    print("  (Lower MSE = the layer can express more complex functions)")
    print()

    torch.manual_seed(42)
    dim = 256
    batch = 512

    W_target = torch.randn(dim, dim) / math.sqrt(dim)
    X = torch.randn(batch, dim)
    Y_target = X @ W_target.T

    models = {
        "Dense (n² params)": DenseLinearLayer(dim),
        "H+Diag (n params)": HadamardDiagonalLayer(dim),
        "Sandwich (2n params)": HadamardSandwichLayer(dim),
    }

    rows = []
    for name, model in models.items():
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()

        for step in range(500):
            pred = model(X)
            loss = F.mse_loss(pred, Y_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            final_loss = F.mse_loss(model(X), Y_target).item()

        rows.append([
            name,
            count_params(model),
            f"{final_loss:.6f}",
        ])

    print_table(rows, ["Layer type", "# Params", "Final MSE (500 steps)"])
    print()
    print("  The sandwich layer (2n params) achieves reasonable approximation")
    print("  of a full dense layer (n² params) — at a tiny fraction of the cost.")
    print("  For feature mixing (not full learning), pure Hadamard is free and effective.")
    print()

    # --- Part D: Memory savings for large models ---
    print("  ─── D. Memory Savings in Large Models ───")
    print()
    print("  If you replace dense mixing layers with Hadamard in a transformer:")
    print()

    rows = []
    for hidden_dim, num_layers in [(768, 12), (1024, 24), (2048, 32), (4096, 48)]:
        dense_total = num_layers * hidden_dim * hidden_dim * 4
        sandwich_total = num_layers * 2 * hidden_dim * 4
        savings = dense_total - sandwich_total

        rows.append([
            f"{hidden_dim}",
            num_layers,
            format_bytes(dense_total),
            format_bytes(sandwich_total),
            format_bytes(savings),
            f"{dense_total / sandwich_total:.0f}x",
        ])

    print_table(rows, ["Hidden dim", "Layers", "Dense memory", "Sandwich memory", "Saved", "Ratio"])
    print()


if __name__ == "__main__":
    run()
