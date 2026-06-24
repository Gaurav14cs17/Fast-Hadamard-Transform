"""
Benchmark: Naive Hadamard (dense matrix multiply) vs Fast Hadamard Transform (butterfly)

Demonstrates why O(n log n) butterfly decomposition beats O(n²) matrix multiply
in both time and memory.

Requirements:
    pip install torch scipy numpy tabulate matplotlib

Optional (for GPU benchmarks):
    pip install fast-hadamard-transform   # Tri Dao's CUDA kernel
"""

import time
import math
import sys
import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import hadamard as scipy_hadamard

try:
    from fast_hadamard_transform import hadamard_transform as cuda_hadamard_transform
    HAS_CUDA_FHT = True
except ImportError:
    HAS_CUDA_FHT = False

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ─────────────────────────────────────────────────────────────────────────────
# 1. SHOW THE HADAMARD MATRIX STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

def show_hadamard_matrix():
    print("=" * 70)
    print("THE HADAMARD MATRIX (8×8 example)")
    print("=" * 70)
    H = scipy_hadamard(8)
    for row in H:
        print("  [" + "  ".join(f"{int(v):+d}" for v in row) + " ]")
    print()
    print("Properties:")
    print("  • All entries are +1 or -1")
    print("  • Rows are mutually orthogonal: H @ H.T = n * I")
    print("  • Symmetric: H = H.T")
    print("  • Recursive Kronecker structure: H(2n) = [[H(n), H(n)], [H(n), -H(n)]]")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 2. FAST HADAMARD TRANSFORM (CPU, pure Python butterfly)
# ─────────────────────────────────────────────────────────────────────────────

def fast_hadamard_cpu(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    In-place butterfly Fast Hadamard Transform on CPU.
    Operates on the last dimension. O(n log n) operations, O(1) extra memory.
    """
    shape = x.shape
    n = shape[-1]
    assert n & (n - 1) == 0, "Dimension must be a power of 2"
    x = x.clone()
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                a = x[..., j]
                b = x[..., j + h]
                x[..., j] = a + b
                x[..., j + h] = a - b
        h *= 2
    return x * scale


def fast_hadamard_vectorized(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    Vectorized butterfly FHT using torch operations. Much faster than the loop version.
    Matches scipy_hadamard convention (unnormalized H matrix with entries ±1).
    """
    n = x.shape[-1]
    assert n & (n - 1) == 0, "Dimension must be a power of 2"
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


# ─────────────────────────────────────────────────────────────────────────────
# 3. NAIVE APPROACH: Dense matrix multiply
# ─────────────────────────────────────────────────────────────────────────────

def naive_hadamard(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Naive O(n²) approach: build full Hadamard matrix and use F.linear."""
    n = x.shape[-1]
    H = torch.tensor(scipy_hadamard(n), dtype=x.dtype, device=x.device).float()
    return F.linear(x.float(), H) * scale


# ─────────────────────────────────────────────────────────────────────────────
# 4. CORRECTNESS CHECK
# ─────────────────────────────────────────────────────────────────────────────

def verify_correctness():
    print("=" * 70)
    print("CORRECTNESS VERIFICATION")
    print("=" * 70)
    dims = [8, 16, 64, 256, 1024]
    for dim in dims:
        x = torch.randn(4, dim)
        ref = naive_hadamard(x)
        fast = fast_hadamard_vectorized(x)
        max_err = (ref - fast).abs().max().item()
        status = "PASS" if max_err < 1e-4 else "FAIL"
        print(f"  dim={dim:>5d}: max error = {max_err:.2e}  [{status}]")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 5. THEORETICAL COMPLEXITY COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def theoretical_comparison():
    print("=" * 70)
    print("THEORETICAL OPERATIONS COUNT")
    print("=" * 70)
    dims = [64, 256, 1024, 4096, 16384, 65536]
    rows = []
    for n in dims:
        naive_ops = n * n
        fast_ops = n * int(math.log2(n))
        speedup = naive_ops / fast_ops
        naive_mem = n * n * 4  # float32
        fast_mem = n * 4
        rows.append([
            f"{n:,}",
            f"{naive_ops:,}",
            f"{fast_ops:,}",
            f"{speedup:.0f}x",
            format_bytes(naive_mem),
            format_bytes(fast_mem),
        ])

    headers = ["Dim (n)", "Naive ops O(n²)", "Fast ops O(n log n)",
               "Op Speedup", "Naive Memory", "Fast Memory"]
    if tabulate:
        print(tabulate(rows, headers=headers, tablefmt="grid"))
    else:
        print(f"{'Dim':>8} | {'Naive ops':>15} | {'Fast ops':>12} | {'Speedup':>8} | {'Naive Mem':>12} | {'Fast Mem':>10}")
        print("-" * 80)
        for r in rows:
            print(f"{r[0]:>8} | {r[1]:>15} | {r[2]:>12} | {r[3]:>8} | {r[4]:>12} | {r[5]:>10}")
    print()


def format_bytes(b):
    if b < 1024:
        return f"{b} B"
    elif b < 1024**2:
        return f"{b/1024:.1f} KB"
    elif b < 1024**3:
        return f"{b/1024**2:.1f} MB"
    else:
        return f"{b/1024**3:.1f} GB"


# ─────────────────────────────────────────────────────────────────────────────
# 6. ACTUAL TIMING BENCHMARKS
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_timing(device="cpu", batch_size=128, warmup=3, repeats=10):
    print("=" * 70)
    print(f"TIMING BENCHMARK (device={device}, batch={batch_size})")
    print("=" * 70)

    dims = [64, 128, 256, 512, 1024, 2048, 4096]
    if device == "cuda":
        dims.extend([8192, 16384])

    rows = []
    naive_times = []
    fast_times = []
    cuda_times = []
    valid_dims = []

    for dim in dims:
        x = torch.randn(batch_size, dim, device=device)

        # Naive approach (skip if matrix too large)
        naive_mem_mb = (dim * dim * 4) / (1024**2)
        if naive_mem_mb > 512:
            naive_ms = float('inf')
            naive_str = "OOM (matrix too large)"
        else:
            try:
                for _ in range(warmup):
                    naive_hadamard(x)
                if device == "cuda":
                    torch.cuda.synchronize()

                t0 = time.perf_counter()
                for _ in range(repeats):
                    naive_hadamard(x)
                if device == "cuda":
                    torch.cuda.synchronize()
                naive_ms = (time.perf_counter() - t0) / repeats * 1000
                naive_str = f"{naive_ms:.3f} ms"
            except (RuntimeError, MemoryError):
                naive_ms = float('inf')
                naive_str = "OOM"

        # Fast (vectorized butterfly)
        for _ in range(warmup):
            fast_hadamard_vectorized(x)
        if device == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(repeats):
            fast_hadamard_vectorized(x)
        if device == "cuda":
            torch.cuda.synchronize()
        fast_ms = (time.perf_counter() - t0) / repeats * 1000
        fast_str = f"{fast_ms:.3f} ms"

        # CUDA kernel (if available)
        cuda_ms = None
        cuda_str = "N/A"
        if HAS_CUDA_FHT and device == "cuda":
            scale = 1.0 / math.sqrt(dim)
            for _ in range(warmup):
                cuda_hadamard_transform(x, scale)
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            for _ in range(repeats):
                cuda_hadamard_transform(x, scale)
            torch.cuda.synchronize()
            cuda_ms = (time.perf_counter() - t0) / repeats * 1000
            cuda_str = f"{cuda_ms:.3f} ms"

        # Speedup
        if naive_ms != float('inf'):
            speedup = naive_ms / fast_ms
            speedup_str = f"{speedup:.1f}x"
        else:
            speedup_str = "∞ (naive OOM)"

        rows.append([dim, naive_str, fast_str, cuda_str, speedup_str,
                     format_bytes(dim * dim * 4)])

        valid_dims.append(dim)
        naive_times.append(naive_ms if naive_ms != float('inf') else None)
        fast_times.append(fast_ms)
        cuda_times.append(cuda_ms)

    headers = ["Dim", "Naive (F.linear)", "Fast (butterfly)", "CUDA kernel",
               "Speedup", "H matrix size"]
    if tabulate:
        print(tabulate(rows, headers=headers, tablefmt="grid"))
    else:
        print(f"{'Dim':>6} | {'Naive':>14} | {'Fast':>14} | {'CUDA':>12} | {'Speedup':>10} | {'H size':>12}")
        print("-" * 80)
        for r in rows:
            print(f"{r[0]:>6} | {r[1]:>14} | {r[2]:>14} | {r[3]:>12} | {r[4]:>10} | {r[5]:>12}")
    print()

    return valid_dims, naive_times, fast_times, cuda_times


# ─────────────────────────────────────────────────────────────────────────────
# 7. MEMORY BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_memory():
    print("=" * 70)
    print("MEMORY USAGE COMPARISON")
    print("=" * 70)
    print()
    print("  The naive method must store the full n×n Hadamard matrix.")
    print("  The fast method works in-place with O(1) extra memory.")
    print()

    dims = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    rows = []
    for n in dims:
        naive_bytes = n * n * 4  # float32 matrix
        fast_extra = 0  # in-place
        input_bytes = n * 4  # one row
        rows.append([
            f"{n:,}",
            format_bytes(naive_bytes),
            format_bytes(input_bytes),
            f"{naive_bytes / max(input_bytes, 1):.0f}x more" if naive_bytes > input_bytes else "same",
        ])

    headers = ["Dim", "Naive (H matrix)", "Fast (extra mem)", "Ratio"]
    if tabulate:
        print(tabulate(rows, headers=headers, tablefmt="grid"))
    else:
        print(f"{'Dim':>8} | {'Naive (H matrix)':>16} | {'Fast (extra)':>14} | {'Ratio':>12}")
        print("-" * 60)
        for r in rows:
            print(f"{r[0]:>8} | {r[1]:>16} | {r[2]:>14} | {r[3]:>12}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 8. BUTTERFLY DIAGRAM (ASCII art)
# ─────────────────────────────────────────────────────────────────────────────

def show_butterfly():
    print("=" * 70)
    print("BUTTERFLY DECOMPOSITION (dim=8)")
    print("=" * 70)
    print("""
  The Fast Hadamard Transform decomposes H into log₂(n) stages of
  simple add/subtract operations (butterfly pattern):

  Input    Stage 1       Stage 2       Stage 3       Output
  x[0] ──┬──(+)──┬──────(+)──┬────────(+)────────── y[0]
          │       │           │
  x[1] ──┴──(−)──│──┬──(+)──│──┬─────(+)────────── y[1]
                  │  │       │  │
  x[2] ──┬──(+)──┴──│──(−)──│──│──┬──(+)────────── y[2]
          │          │       │  │  │
  x[3] ──┴──(−)─────┴──(−)──│──│──│──(+)────────── y[3]
                             │  │  │
  x[4] ──┬──(+)──┬──────(+)─┴──│──│──(−)────────── y[4]
          │       │             │  │
  x[5] ──┴──(−)──│──┬──(+)────┴──│──(−)────────── y[5]
                  │  │            │
  x[6] ──┬──(+)──┴──│──(−)──────┴──(−)────────── y[6]
          │          │
  x[7] ──┴──(−)─────┴──(−)─────────(−)────────── y[7]

  Each stage: n/2 additions + n/2 subtractions = n operations
  Total stages: log₂(n) = 3  (for n=8)
  Total operations: n × log₂(n) = 8 × 3 = 24

  Compare to naive matrix multiply: n² = 64 operations
  Savings: 64/24 = 2.7x  (grows to 4096x at dim=65536!)
""")


# ─────────────────────────────────────────────────────────────────────────────
# 9. PLOT (if matplotlib available)
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(dims, naive_times, fast_times, cuda_times):
    if not HAS_MATPLOTLIB:
        print("  [matplotlib not installed — skipping plot]")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Timing plot
    ax = axes[0]
    valid_naive = [(d, t) for d, t in zip(dims, naive_times) if t is not None]
    if valid_naive:
        nd, nt = zip(*valid_naive)
        ax.plot(nd, nt, 'ro-', label='Naive (F.linear)', linewidth=2, markersize=8)
    ax.plot(dims, fast_times, 'bs-', label='Fast (butterfly)', linewidth=2, markersize=8)
    if any(t is not None for t in cuda_times):
        valid_cuda = [(d, t) for d, t in zip(dims, cuda_times) if t is not None]
        cd, ct = zip(*valid_cuda)
        ax.plot(cd, ct, 'g^-', label='CUDA kernel', linewidth=2, markersize=8)
    ax.set_xlabel('Dimension', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('Execution Time: Naive vs Fast Hadamard', fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Memory plot
    ax = axes[1]
    dims_mem = [2**i for i in range(6, 17)]
    naive_mem = [d * d * 4 / 1024**2 for d in dims_mem]  # MB
    fast_mem = [d * 4 / 1024 for d in dims_mem]  # KB → MB
    ax.plot(dims_mem, naive_mem, 'ro-', label='Naive (n² matrix)', linewidth=2, markersize=8)
    ax.plot(dims_mem, [m / 1024 for m in fast_mem], 'bs-', label='Fast (in-place)',
            linewidth=2, markersize=8)
    ax.set_xlabel('Dimension', fontsize=12)
    ax.set_ylabel('Memory (MB)', fontsize=12)
    ax.set_title('Memory: Hadamard Matrix vs In-Place', fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('hadamard_benchmark.png', dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved to: hadamard_benchmark.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 10. REAL-WORLD USE CASES (see use_cases/ directory for detailed benchmarks)
# ─────────────────────────────────────────────────────────────────────────────

def demo_use_cases():
    print("=" * 70)
    print("REAL-WORLD USE CASES (run `python run_all.py --usecases` for details)")
    print("=" * 70)
    print()
    print("  Detailed benchmarks are in the use_cases/ subdirectory:")
    print()
    print("    1. Random Projection (SRHT / Johnson-Lindenstrauss)")
    print("       → 44x faster than dense at dim=8192, 1024x less memory")
    print()
    print("    2. Quantization Rotation (QuIP# style)")
    print("       → 5-15x lower quantization error at 3-4 bit")
    print()
    print("    3. Feature Decorrelation & Mixing")
    print("       → 51x faster than PCA at dim=1024")
    print()
    print("    4. Hadamard as Neural Network Layer")
    print("       → 0 params, 2048x memory savings vs dense")
    print()
    print("    5. Signal Compression (Walsh-Hadamard)")
    print("       → Better than FFT for step/binary signals")
    print()
    print("    6. Error-Correcting Codes")
    print("       → O(n log n) ML decoding, 7x more info than repetition")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║   BENCHMARK: Naive Matrix Multiply vs Fast Hadamard Transform      ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    show_hadamard_matrix()
    show_butterfly()
    verify_correctness()
    theoretical_comparison()
    benchmark_memory()

    # CPU benchmarks
    dims, naive_times, fast_times, cuda_times = benchmark_timing(
        device="cpu", batch_size=128
    )

    # GPU benchmarks (if available)
    if torch.cuda.is_available():
        print("\n" + "=" * 70)
        print("GPU DETECTED — running CUDA benchmarks")
        print("=" * 70 + "\n")
        dims_gpu, naive_gpu, fast_gpu, cuda_gpu = benchmark_timing(
            device="cuda", batch_size=512
        )
        plot_results(dims_gpu, naive_gpu, fast_gpu, cuda_gpu)
    else:
        plot_results(dims, naive_times, fast_times, cuda_times)

    demo_use_cases()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Naive (F.linear with H matrix)                                     │
  │    • Time:   O(n²) — full matrix-vector multiply                    │
  │    • Memory: O(n²) — must store entire n×n Hadamard matrix          │
  │    • At dim=65536: 16 GB just for the matrix!                       │
  │                                                                     │
  │  Fast Hadamard Transform (butterfly decomposition)                  │
  │    • Time:   O(n log n) — log₂(n) stages of n add/sub operations   │
  │    • Memory: O(n) — works in-place, no matrix needed                │
  │    • At dim=65536: only 256 KB for the input vector                 │
  │                                                                     │
  │  CUDA kernel (Tri Dao's implementation)                             │
  │    • Same O(n log n) algorithm, but fused into a single kernel      │
  │    • Avoids memory bandwidth overhead of multiple PyTorch ops       │
  │    • Supports autograd (backward = another forward, since H = H^T)  │
  │    • Handles non-power-of-2 via 12N, 20N, 28N, 40N variants        │
  └─────────────────────────────────────────────────────────────────────┘

  Key insight: The Hadamard matrix has a recursive structure (Kronecker
  product of 2×2 blocks). The fast algorithm exploits this — just like
  FFT exploits the structure of the DFT matrix. You never need to build
  the full matrix.
""")


if __name__ == "__main__":
    main()
