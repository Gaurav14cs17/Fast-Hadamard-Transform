"""
USE CASE 6: Error-Correcting Codes (Reed-Muller / Hadamard Codes)

Problem: Transmit data reliably over a noisy channel.
Solution: Encode data using Hadamard matrix rows as codewords.

Hadamard codes are a family of error-correcting codes where:
  - Codewords are rows (or columns) of the Hadamard matrix
  - Minimum distance = n/2 (corrects up to n/4 - 1 errors)
  - Decoding uses the Fast Hadamard Transform for O(n log n) ML decoding

This is one of the oldest applications of Hadamard matrices (1960s, Mariner missions).
"""

import time
import math
import torch
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from use_cases.common import (
    fast_hadamard_vectorized, format_bytes, print_table,
    time_fn, section_header
)
from scipy.linalg import hadamard as scipy_hadamard


def hadamard_encode(message_bits, n):
    """
    Encode a message index (0 to n-1) as a Hadamard codeword.
    The codeword is the message_idx-th row of H mapped to ±1.
    """
    H = torch.tensor(scipy_hadamard(n), dtype=torch.float32)
    codewords = H[message_bits]  # shape: (batch, n), values ±1
    return codewords


def hadamard_decode_naive(received, n):
    """Naive ML decoding: correlate with all codewords. O(n²)."""
    H = torch.tensor(scipy_hadamard(n), dtype=torch.float32)
    correlations = received @ H.T  # (batch, n)
    decoded = correlations.abs().argmax(dim=1)
    signs = torch.sign(correlations.gather(1, decoded.unsqueeze(1))).squeeze(1)
    return decoded, signs


def hadamard_decode_fast(received, n):
    """Fast ML decoding using FHT. O(n log n)."""
    correlations = fast_hadamard_vectorized(received)  # equivalent to received @ H
    decoded = correlations.abs().argmax(dim=1)
    signs = torch.sign(correlations.gather(1, decoded.unsqueeze(1))).squeeze(1)
    return decoded, signs


def add_noise(codewords, error_rate):
    """Flip bits (±1 → ∓1) with given probability."""
    noise_mask = torch.rand_like(codewords) < error_rate
    noisy = codewords.clone()
    noisy[noise_mask] *= -1
    return noisy


def run():
    section_header("USE CASE 6: Error-Correcting Codes (Hadamard Codes)")

    print("  Hadamard codes: rows of H are codewords with distance n/2.")
    print("  Fast decoding via FHT replaces O(n²) correlation with O(n log n).")
    print()

    # --- Part A: Code properties ---
    print("  ─── A. Hadamard Code Properties ───")
    print()

    rows = []
    for log_n in range(3, 11):
        n = 2 ** log_n
        k_bits = log_n + 1  # can encode log2(2n) = log2(n) + 1 bits (with sign)
        rate = k_bits / n
        min_dist = n // 2
        correctable = min_dist // 2 - 1

        rows.append([
            n,
            k_bits,
            f"{rate:.4f}",
            min_dist,
            correctable,
            f"{correctable/n*100:.1f}%",
        ])

    print_table(rows, ["Code length n", "Message bits", "Rate", "Min distance", "Correctable errors", "Error tolerance"])
    print()

    # --- Part B: Decoding accuracy vs noise ---
    print("  ─── B. Decoding Accuracy vs Channel Error Rate ───")
    print()

    n = 64
    num_messages = 1000
    error_rates = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

    torch.manual_seed(42)
    messages = torch.randint(0, n, (num_messages,))
    codewords = hadamard_encode(messages, n)

    rows = []
    for err_rate in error_rates:
        received = add_noise(codewords, err_rate)
        decoded_idx, decoded_sign = hadamard_decode_fast(received, n)
        accuracy = (decoded_idx == messages).float().mean().item()
        num_flipped = int(err_rate * n)

        rows.append([
            f"{err_rate*100:.0f}%",
            f"~{num_flipped}/{n}",
            f"{accuracy*100:.1f}%",
            "OK" if accuracy > 0.99 else ("Degraded" if accuracy > 0.5 else "Failed"),
        ])

    print(f"  Code: n={n}, can correct up to {n//4 - 1} errors per codeword")
    print()
    print_table(rows, ["Error rate", "Flipped bits", "Decode accuracy", "Status"])
    print()

    # --- Part C: Decoding speed comparison ---
    print("  ─── C. Decoding Speed: Naive O(n²) vs Fast O(n log n) ───")
    print()

    rows = []
    for log_n in range(4, 11):
        n = 2 ** log_n
        batch = 256
        received = torch.randn(batch, n)  # soft received values

        t_naive, _ = time_fn(hadamard_decode_naive, received, n, warmup=3, repeats=10)
        t_fast, _ = time_fn(hadamard_decode_fast, received, n, warmup=3, repeats=10)
        speedup = t_naive / t_fast

        rows.append([
            n,
            batch,
            f"{t_naive:.3f} ms",
            f"{t_fast:.3f} ms",
            f"{speedup:.1f}x",
        ])

    print_table(rows, ["Code length", "Batch", "Naive decode", "Fast decode (FHT)", "Speedup"])
    print()

    # --- Part D: Comparison with repetition code ---
    print("  ─── D. Hadamard Code vs Repetition Code ───")
    print()
    print("  Both use n channel symbols. Which encodes more information reliably?")
    print()

    n = 64
    err_rate = 0.15
    num_trials = 2000
    torch.manual_seed(0)

    # Hadamard code: encodes log2(n)+1 bits in n symbols
    messages_h = torch.randint(0, n, (num_trials,))
    codewords_h = hadamard_encode(messages_h, n)
    received_h = add_noise(codewords_h, err_rate)
    decoded_h, _ = hadamard_decode_fast(received_h, n)
    acc_hadamard = (decoded_h == messages_h).float().mean().item()
    bits_hadamard = int(math.log2(n)) + 1

    # Repetition code: encodes 1 bit in n symbols
    messages_r = torch.randint(0, 2, (num_trials,)).float() * 2 - 1  # ±1
    codewords_r = messages_r.unsqueeze(1).expand(-1, n)
    received_r = add_noise(codewords_r, err_rate)
    decoded_r = torch.sign(received_r.sum(dim=1))
    acc_rep = (decoded_r == messages_r).float().mean().item()
    bits_rep = 1

    rows = [
        ["Hadamard code", n, bits_hadamard, f"{bits_hadamard/n:.4f}", f"{acc_hadamard*100:.1f}%"],
        ["Repetition code", n, bits_rep, f"{bits_rep/n:.4f}", f"{acc_rep*100:.1f}%"],
    ]

    print_table(rows, ["Code", "Length n", "Info bits", "Rate", "Accuracy @15% err"])
    print()
    print(f"  Hadamard code transmits {bits_hadamard}x more information at similar reliability!")
    print()


if __name__ == "__main__":
    run()
