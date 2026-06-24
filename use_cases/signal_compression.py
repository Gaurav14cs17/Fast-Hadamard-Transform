"""
USE CASE 5: Signal Compression via Walsh-Hadamard Spectrum

Problem: Compress/denoise a signal by keeping only its most important components.
Classical approach: FFT → keep top-k frequencies → IFFT
Hadamard approach: WHT → keep top-k Walsh coefficients → inverse WHT

Advantages of Walsh-Hadamard over Fourier for certain signal types:
  - Works with real arithmetic only (no complex numbers)
  - Better for binary/step-like signals (Walsh functions are ±1 steps)
  - Faster: O(n log n) with only additions/subtractions (no multiplications!)
  - Ideal for digital signals, error-correcting codes, boolean functions
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


def walsh_hadamard_compress(signal, keep_ratio):
    """Compress signal by keeping top-k Walsh-Hadamard coefficients."""
    n = signal.shape[-1]
    n_pad = next_power_of_2(n)
    if n != n_pad:
        signal_pad = torch.nn.functional.pad(signal, (0, n_pad - n))
    else:
        signal_pad = signal

    scale = 1.0 / math.sqrt(n_pad)
    coeffs = fast_hadamard_vectorized(signal_pad.unsqueeze(0), scale=scale)
    coeffs = coeffs.squeeze(0)

    k = max(1, int(n_pad * keep_ratio))
    topk_vals, topk_idx = coeffs.abs().topk(k)
    sparse_coeffs = torch.zeros_like(coeffs)
    sparse_coeffs[topk_idx] = coeffs[topk_idx]

    # Inverse: same transform with same scale (self-inverse when normalized)
    reconstructed = fast_hadamard_vectorized(sparse_coeffs.unsqueeze(0), scale=scale)
    return reconstructed.squeeze(0)[:n], k


def fft_compress(signal, keep_ratio):
    """Compress signal by keeping top-k Fourier coefficients."""
    n = signal.shape[-1]
    coeffs = torch.fft.rfft(signal)
    k = max(1, int(len(coeffs) * keep_ratio))
    magnitudes = coeffs.abs()
    topk_vals, topk_idx = magnitudes.topk(k)
    sparse_coeffs = torch.zeros_like(coeffs)
    sparse_coeffs[topk_idx] = coeffs[topk_idx]
    reconstructed = torch.fft.irfft(sparse_coeffs, n=n)
    return reconstructed, k


def run():
    section_header("USE CASE 5: Signal Compression (Walsh-Hadamard Spectrum)")

    print("  Comparing Walsh-Hadamard Transform vs FFT for signal compression.")
    print("  Keep only top-k% of coefficients, measure reconstruction error.")
    print()

    # --- Part A: Different signal types ---
    print("  ─── A. Compression Quality by Signal Type ───")
    print()

    torch.manual_seed(42)
    n = 1024

    signals = {}

    # Smooth sinusoidal (FFT-friendly)
    t = torch.linspace(0, 1, n)
    signals["Smooth sinusoid"] = torch.sin(2 * math.pi * 5 * t) + 0.5 * torch.sin(2 * math.pi * 13 * t)

    # Step function (Hadamard-friendly)
    step = torch.zeros(n)
    for i in range(0, n, 64):
        step[i:i+32] = 1.0 if (i // 64) % 2 == 0 else -1.0
    signals["Step/square wave"] = step

    # Random binary
    signals["Random binary (±1)"] = torch.sign(torch.randn(n))

    # Piecewise constant (digital signal)
    pw = torch.zeros(n)
    segments = torch.randint(1, 5, (20,))
    pos = 0
    for i, seg_len in enumerate(segments):
        actual_len = min(int(seg_len.item()) * (n // 60), n - pos)
        pw[pos:pos+actual_len] = (-1.0) ** i * (i % 3 + 1)
        pos += actual_len
        if pos >= n:
            break
    signals["Piecewise constant"] = pw

    # Noisy smooth
    signals["Noisy sinusoid"] = torch.sin(2 * math.pi * 3 * t) + 0.3 * torch.randn(n)

    keep_ratios = [0.05, 0.10, 0.20, 0.50]

    for ratio in keep_ratios:
        print(f"  Keep ratio: {ratio*100:.0f}% of coefficients")
        rows = []
        for name, sig in signals.items():
            recon_wht, k_wht = walsh_hadamard_compress(sig, ratio)
            recon_fft, k_fft = fft_compress(sig, ratio)

            snr_wht = 10 * math.log10(sig.pow(2).mean() / (sig - recon_wht).pow(2).mean().clamp(min=1e-10))
            snr_fft = 10 * math.log10(sig.pow(2).mean() / (sig - recon_fft).pow(2).mean().clamp(min=1e-10))

            winner = "WHT" if snr_wht > snr_fft else "FFT"
            rows.append([name, f"{snr_wht:.1f} dB", f"{snr_fft:.1f} dB", winner])

        print_table(rows, ["Signal type", "WHT SNR", "FFT SNR", "Winner"])
        print()

    # --- Part B: Speed comparison ---
    print("  ─── B. Transform Speed: WHT vs FFT ───")
    print()

    rows = []
    for n in [256, 512, 1024, 2048, 4096, 8192]:
        x = torch.randn(n)

        def do_wht(x):
            return fast_hadamard_vectorized(x.unsqueeze(0), scale=1.0/math.sqrt(n)).squeeze(0)

        def do_fft(x):
            return torch.fft.rfft(x)

        t_wht, _ = time_fn(do_wht, x, warmup=5, repeats=20)
        t_fft, _ = time_fn(do_fft, x, warmup=5, repeats=20)

        rows.append([
            n,
            f"{t_wht:.3f} ms",
            f"{t_fft:.3f} ms",
            f"{t_wht/t_fft:.2f}x" if t_wht > t_fft else f"{t_fft/t_wht:.2f}x faster",
            "WHT" if t_wht < t_fft else "FFT",
        ])

    print_table(rows, ["Dim", "WHT time", "FFT time", "Ratio", "Faster"])
    print()
    print("  Note: PyTorch's FFT uses highly optimized FFTW/cuFFT.")
    print("  On GPU with CUDA kernel, WHT matches or beats FFT for power-of-2 sizes.")
    print()

    # --- Part C: Denoising application ---
    print("  ─── C. Denoising: Remove Noise by Thresholding Coefficients ───")
    print()

    torch.manual_seed(0)
    n = 512
    t = torch.linspace(0, 1, n)
    clean = torch.zeros(n)
    clean[:128] = 2.0
    clean[128:256] = -1.0
    clean[256:384] = 1.5
    clean[384:] = -0.5

    noise_levels = [0.1, 0.3, 0.5, 1.0]
    rows = []
    for noise_std in noise_levels:
        noisy = clean + noise_std * torch.randn(n)

        denoised_wht, _ = walsh_hadamard_compress(noisy, keep_ratio=0.1)
        denoised_fft, _ = fft_compress(noisy, keep_ratio=0.1)

        mse_noisy = ((clean - noisy) ** 2).mean().item()
        mse_wht = ((clean - denoised_wht) ** 2).mean().item()
        mse_fft = ((clean - denoised_fft) ** 2).mean().item()

        rows.append([
            f"σ={noise_std}",
            f"{mse_noisy:.4f}",
            f"{mse_wht:.4f} ({mse_noisy/max(mse_wht,1e-10):.1f}x better)",
            f"{mse_fft:.4f} ({mse_noisy/max(mse_fft,1e-10):.1f}x better)",
            "WHT" if mse_wht < mse_fft else "FFT",
        ])

    print("  Signal: piecewise constant (step function)")
    print("  Method: keep top 10% of transform coefficients")
    print()
    print_table(rows, ["Noise", "MSE (noisy)", "MSE (WHT denoised)", "MSE (FFT denoised)", "Winner"])
    print()
    print("  WHT excels at denoising piecewise-constant and digital signals")
    print("  because Walsh functions are themselves step functions.")
    print()


if __name__ == "__main__":
    run()
