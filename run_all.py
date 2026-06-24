"""
Fast Hadamard Transform — Complete Benchmark & Use Cases
=========================================================

Runs the core benchmark (naive vs fast) and all real-world use case demos.

Usage:
    python run_all.py              # Run everything
    python run_all.py --core       # Core benchmark only
    python run_all.py --usecases   # Use cases only
    python run_all.py --case 1     # Specific use case (1-6)

Structure:
    benchmark_hadamard.py          Core timing/memory benchmark
    use_cases/
      ├── random_projection.py     UC1: SRHT / Johnson-Lindenstrauss
      ├── quantization_rotation.py UC2: QuIP#-style weight rotation
      ├── feature_hashing.py       UC3: Feature decorrelation & mixing
      ├── neural_network_layer.py  UC4: Hadamard as cheap linear layer
      ├── signal_compression.py    UC5: Walsh-Hadamard spectral compression
      └── error_correcting_codes.py UC6: Hadamard codes for error correction
    results/                       Output plots and saved data
"""

import sys
import time

def banner():
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║     Fast Hadamard Transform — Benchmark & Use Cases Suite          ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print("║  Core: Naive O(n²) matrix vs Fast O(n log n) butterfly             ║")
    print("║  + 6 real-world application benchmarks with data                   ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()


def run_core():
    print("\n" + "▓" * 70)
    print("▓  CORE BENCHMARK: Naive Matrix Multiply vs Fast Hadamard Transform")
    print("▓" * 70 + "\n")
    from benchmark_hadamard import main as core_main
    core_main()


def run_use_case(num):
    modules = {
        1: ("use_cases.random_projection", "Random Projection (SRHT / JL)"),
        2: ("use_cases.quantization_rotation", "Quantization Rotation (QuIP#)"),
        3: ("use_cases.feature_hashing", "Feature Decorrelation & Mixing"),
        4: ("use_cases.neural_network_layer", "Hadamard as Neural Network Layer"),
        5: ("use_cases.signal_compression", "Signal Compression (Walsh-Hadamard)"),
        6: ("use_cases.error_correcting_codes", "Error-Correcting Codes"),
    }

    if num not in modules:
        print(f"  Error: Use case {num} not found. Available: 1-6")
        return

    mod_path, title = modules[num]
    print(f"\n{'▓' * 70}")
    print(f"▓  USE CASE {num}: {title}")
    print(f"{'▓' * 70}\n")

    import importlib
    mod = importlib.import_module(mod_path)
    mod.run()


def run_all_use_cases():
    for i in range(1, 7):
        run_use_case(i)


def print_summary():
    print()
    print("═" * 70)
    print("  COMPLETE SUMMARY: Why Use Fast Hadamard Transform?")
    print("═" * 70)
    print("""
  ┌────────────────────────────────────────────────────────────────────┐
  │                                                                    │
  │  USE CASE                    WHY FHT HELPS                         │
  │  ─────────────────────────── ────────────────────────────────────  │
  │  1. Random Projection        O(n log n) vs O(nk) dense multiply   │
  │     (SRHT / FastJL)          Memory: O(n) vs O(nk) for R matrix   │
  │                                                                    │
  │  2. Quantization Rotation    Spreads outliers → lower quant error  │
  │     (QuIP#, GPTQ variants)   Rotation is free (H = H^T = H^-1)   │
  │                                                                    │
  │  3. Feature Decorrelation    O(n log n) vs O(n³) for PCA          │
  │     (Whitening substitute)   No data-dependent computation        │
  │                                                                    │
  │  4. Neural Network Layer     0 params vs n² for dense linear      │
  │     (FNet, Monarch, mixing)  Sandwich H·D·H ≈ full linear         │
  │                                                                    │
  │  5. Signal Compression       Better than FFT for step signals     │
  │     (Walsh-Hadamard domain)  Real arithmetic only (no complex)    │
  │                                                                    │
  │  6. Error-Correcting Codes   O(n log n) ML decoding vs O(n²)      │
  │     (Hadamard / Reed-Muller) Maximum distance codes               │
  │                                                                    │
  ├────────────────────────────────────────────────────────────────────┤
  │                                                                    │
  │  COMMON THREAD: The Hadamard matrix has Kronecker structure        │
  │  that enables O(n log n) computation via butterfly operations.     │
  │  Tri Dao's CUDA kernel fuses this into a single GPU pass.          │
  │                                                                    │
  └────────────────────────────────────────────────────────────────────┘
""")


def main():
    banner()

    args = sys.argv[1:]

    if not args:
        t0 = time.time()
        run_core()
        run_all_use_cases()
        print_summary()
        elapsed = time.time() - t0
        print(f"  Total runtime: {elapsed:.1f}s")
        print()
    elif "--core" in args:
        run_core()
    elif "--usecases" in args:
        run_all_use_cases()
        print_summary()
    elif "--case" in args:
        idx = args.index("--case")
        if idx + 1 < len(args):
            case_num = int(args[idx + 1])
            run_use_case(case_num)
        else:
            print("Usage: python run_all.py --case <1-6>")
    else:
        print("Usage:")
        print("  python run_all.py              # Run everything")
        print("  python run_all.py --core       # Core benchmark only")
        print("  python run_all.py --usecases   # All use cases")
        print("  python run_all.py --case N     # Specific use case (1-6)")


if __name__ == "__main__":
    main()
