"""
USE CASE 2: Weight Rotation for Quantization (QuIP# style)

Problem: When quantizing neural network weights to low-bit (e.g. 4-bit, 2-bit),
outlier values cause massive quantization error.

Solution: Multiply weights by a Hadamard matrix before quantizing. This "rotates"
the weight distribution, spreading outliers across all dimensions, making the
distribution more uniform and quantization-friendly.

Key properties:
  - Hadamard rotation is orthogonal → preserves norms (no information loss)
  - Spreads concentrated energy uniformly across dimensions
  - De-rotation after inference is just another Hadamard multiply (free!)
  - The Fast Hadamard Transform makes this practical for large models
"""

import time
import math
import torch
import torch.nn.functional as F
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from use_cases.common import (
    fast_hadamard_vectorized, format_bytes, print_table,
    time_fn, section_header, next_power_of_2
)


def uniform_quantize(x, bits):
    """Symmetric uniform quantization to `bits` bits."""
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().max() / qmax
    x_q = torch.clamp(torch.round(x / scale), -qmax, qmax)
    return x_q * scale, scale


def per_channel_quantize(x, bits):
    """Per-channel (per-row) symmetric quantization."""
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().amax(dim=-1, keepdim=True) / qmax
    scale = scale.clamp(min=1e-8)
    x_q = torch.clamp(torch.round(x / scale), -qmax, qmax)
    return x_q * scale, scale


def quantize_with_hadamard_rotation(W, bits):
    """Rotate weights with Hadamard, then quantize, then de-rotate."""
    n = W.shape[-1]
    n_pad = next_power_of_2(n)
    if n != n_pad:
        W_pad = F.pad(W, (0, n_pad - n))
    else:
        W_pad = W

    # Forward: H @ W / sqrt(n) — normalized orthogonal transform
    scale = 1.0 / math.sqrt(n_pad)
    W_rotated = fast_hadamard_vectorized(W_pad, scale=scale)
    W_quantized, q_scale = per_channel_quantize(W_rotated, bits)
    # Inverse: H @ W_q / sqrt(n) — since H/sqrt(n) is its own inverse
    W_recovered = fast_hadamard_vectorized(W_quantized, scale=scale)

    if n != n_pad:
        W_recovered = W_recovered[..., :n]
    return W_recovered


def run():
    section_header("USE CASE 2: Weight Rotation for Quantization (QuIP# style)")

    print("  Simulating quantization of neural network weight matrices.")
    print("  Comparing: direct quantization vs Hadamard-rotated quantization.")
    print()

    # --- Part A: Visualize distribution change ---
    print("  ─── A. How Rotation Changes the Weight Distribution ───")
    print()

    torch.manual_seed(42)
    n = 256
    W = torch.randn(64, n)
    W[:, 0] *= 10  # inject outlier column (common in real LLMs)
    W[:, 1] *= 8

    n_pad = next_power_of_2(n)
    W_rotated = fast_hadamard_vectorized(W, scale=1.0 / math.sqrt(n_pad))

    print(f"  Original weights (with outlier columns):")
    print(f"    max |w|:   {W.abs().max().item():.4f}")
    print(f"    std:       {W.std().item():.4f}")
    print(f"    kurtosis:  {((W - W.mean()) ** 4).mean().item() / W.std().item()**4:.2f}")
    print(f"    max/mean:  {W.abs().max().item() / W.abs().mean().item():.2f}")
    print()
    print(f"  After Hadamard rotation:")
    print(f"    max |w|:   {W_rotated.abs().max().item():.4f}")
    print(f"    std:       {W_rotated.std().item():.4f}")
    print(f"    kurtosis:  {((W_rotated - W_rotated.mean()) ** 4).mean().item() / W_rotated.std().item()**4:.2f}")
    print(f"    max/mean:  {W_rotated.abs().max().item() / W_rotated.abs().mean().item():.2f}")
    print()
    print("  → Rotation reduces kurtosis and max/mean ratio → better for quantization")
    print()

    # --- Part B: Quantization error comparison ---
    print("  ─── B. Quantization Error (MSE) at Different Bit Widths ───")
    print()

    sizes = [(128, 256), (256, 512), (512, 1024), (768, 768)]
    bit_widths = [2, 3, 4, 8]

    rows = []
    for out_dim, in_dim in sizes:
        torch.manual_seed(0)
        W = torch.randn(out_dim, in_dim)
        W[:, :4] *= 8  # inject outliers

        for bits in bit_widths:
            W_direct, _ = per_channel_quantize(W, bits)
            W_rotated_q = quantize_with_hadamard_rotation(W, bits)

            mse_direct = ((W - W_direct) ** 2).mean().item()
            mse_rotated = ((W - W_rotated_q) ** 2).mean().item()
            improvement = mse_direct / max(mse_rotated, 1e-10)

            rows.append([
                f"{out_dim}×{in_dim}",
                f"{bits}-bit",
                f"{mse_direct:.6f}",
                f"{mse_rotated:.6f}",
                f"{improvement:.2f}x better",
            ])

    print_table(rows, ["Weight shape", "Bits", "MSE (direct)", "MSE (rotated)", "Improvement"])
    print()

    # --- Part C: Impact on inference accuracy ---
    print("  ─── C. Impact on Layer Output (Simulated Inference) ───")
    print()
    print("  Measuring output error: ||W·x - W_quantized·x|| / ||W·x||")
    print()

    rows = []
    for out_dim, in_dim in [(256, 512), (512, 1024), (1024, 2048)]:
        torch.manual_seed(1)
        W = torch.randn(out_dim, in_dim)
        W[:, :8] *= 6
        x = torch.randn(32, in_dim)
        y_true = x @ W.T

        for bits in [4, 3, 2]:
            W_direct, _ = per_channel_quantize(W, bits)
            y_direct = x @ W_direct.T
            err_direct = torch.norm(y_true - y_direct) / torch.norm(y_true)

            W_rot_q = quantize_with_hadamard_rotation(W, bits)
            y_rot = x @ W_rot_q.T
            err_rot = torch.norm(y_true - y_rot) / torch.norm(y_true)

            rows.append([
                f"{out_dim}×{in_dim}",
                f"{bits}-bit",
                f"{err_direct.item():.4f}",
                f"{err_rot.item():.4f}",
                f"{err_direct.item() / max(err_rot.item(), 1e-10):.2f}x",
            ])

    print_table(rows, ["Layer shape", "Bits", "Rel. Error (direct)", "Rel. Error (rotated)", "Improvement"])
    print()

    # --- Part D: Timing overhead ---
    print("  ─── D. Timing Overhead of Hadamard Rotation ───")
    print()
    print("  Is the rotation overhead worth it?")
    print()

    rows = []
    for out_dim, in_dim in [(256, 256), (512, 512), (1024, 1024), (2048, 2048)]:
        W = torch.randn(out_dim, in_dim)
        bits = 4

        t_direct, _ = time_fn(per_channel_quantize, W, bits)
        t_rotated, _ = time_fn(quantize_with_hadamard_rotation, W, bits)
        overhead = (t_rotated - t_direct) / t_direct * 100

        rows.append([
            f"{out_dim}×{in_dim}",
            f"{t_direct:.3f} ms",
            f"{t_rotated:.3f} ms",
            f"+{overhead:.0f}%",
            "Worth it (huge quality gain)" if overhead < 200 else "Borderline",
        ])

    print_table(rows, ["Shape", "Direct quant", "Rotated quant", "Overhead", "Verdict"])
    print()
    print("  The rotation overhead is minimal compared to the quality improvement,")
    print("  especially at low bit-widths (2-4 bit) where outliers are devastating.")
    print()


if __name__ == "__main__":
    run()
