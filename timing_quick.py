#!/usr/bin/env python
"""Quick timing test for Mamba2Chunked with torch.compile support."""

import sys
import time
import torch
import os
from contextlib import contextmanager

sys.path.insert(0, '/home/tzeng/repos/chunk_mamba')

from mamba_ssm.modules.mamba2_chunked import Mamba2Chunked


@contextmanager
def suppress_torch_compile_output():
    """Suppress verbose torch.compile output."""
    import logging
    # Save current settings
    old_log_level = logging.getLogger("torch._dynamo").level
    old_verbose = os.environ.get("TORCH_LOGS", None)

    # Suppress output
    logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
    logging.getLogger("torch._inductor").setLevel(logging.ERROR)
    os.environ["TORCH_LOGS"] = ""

    try:
        yield
    finally:
        # Restore settings
        logging.getLogger("torch._dynamo").setLevel(old_log_level)
        logging.getLogger("torch._inductor").setLevel(old_log_level)
        if old_verbose is not None:
            os.environ["TORCH_LOGS"] = old_verbose
        else:
            os.environ.pop("TORCH_LOGS", None)


def time_forward(model, u, n_warmup=5, n_runs=20):
    """Time forward pass with warmup."""
    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model(u)

    # Benchmark
    torch.cuda.synchronize()
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(u)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    times = torch.tensor(times)
    return times.mean().item() * 1000, times.std().item() * 1000


def time_backward(model, u, n_warmup=3, n_runs=10):
    """Time forward + backward pass with warmup."""
    # Warmup
    for _ in range(n_warmup):
        u_copy = u.clone().requires_grad_(True)
        y = model(u_copy)
        loss = y.sum()
        loss.backward()

    # Benchmark
    torch.cuda.synchronize()
    times = []
    for _ in range(n_runs):
        u_copy = u.clone().requires_grad_(True)

        start = time.perf_counter()
        y = model(u_copy)
        loss = y.sum()
        loss.backward()
        torch.cuda.synchronize()

        times.append(time.perf_counter() - start)

    times = torch.tensor(times)
    return times.mean().item() * 1000, times.std().item() * 1000


def main():
    print("Comprehensive Timing Test - Mamba2Chunked")
    print("="*70)

    torch.manual_seed(42)

    # Config
    batch, seqlen, d_model = 2, 1024, 512
    chunk_size = 256

    print(f"Config: batch={batch}, seqlen={seqlen}, d_model={d_model}")
    print(f"Chunk size: {chunk_size}")
    print(f"Total tokens: {batch * seqlen:,}")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print()

    # Create model
    model = Mamba2Chunked(
        d_model=d_model,
        d_state=64,
        d_conv=4,
        expand=2,
        headdim=64,
        ngroups=1,
        chunk_size=chunk_size,
        device="cuda",
        dtype=torch.float32,
    ).cuda()

    # Initialize R matrices with small values
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.startswith("R_"):
                p.normal_(0, 0.01)

    u = torch.randn(batch, seqlen, d_model, device="cuda", dtype=torch.float32)

    results = {}

    # ========================================================================
    # FORWARD PASS BENCHMARKS
    # ========================================================================
    print("-"*70)
    print("FORWARD PASS (inference)")
    print("-"*70)

    model.eval()

    # 1. Naive PyTorch
    print("\n[1] Naive PyTorch (eager)...")
    model.use_triton = False
    mean, std = time_forward(model, u, n_warmup=5, n_runs=20)
    tput = (batch * seqlen) / (mean / 1000)
    print(f"    {mean:7.2f} ± {std:5.2f} ms  |  {tput:>10,.0f} tokens/sec")
    results['naive_fwd'] = mean

    # 2. Naive + torch.compile
    print("\n[2] Naive PyTorch + torch.compile...")
    print("    Compiling model (this may take 30-60 seconds)...", end='', flush=True)
    model.use_triton = False
    try:
        with suppress_torch_compile_output():
            model_compiled = torch.compile(model, mode='reduce-overhead', fullgraph=False)
            mean, std = time_forward(model_compiled, u, n_warmup=10, n_runs=20)
        print(" done!")
        tput = (batch * seqlen) / (mean / 1000)
        print(f"    {mean:7.2f} ± {std:5.2f} ms  |  {tput:>10,.0f} tokens/sec")
        results['naive_compiled_fwd'] = mean
        del model_compiled
        torch.cuda.empty_cache()
    except Exception as e:
        print(" FAILED!")
        print(f"    torch.compile error: {type(e).__name__}: {str(e)[:60]}")
        results['naive_compiled_fwd'] = results['naive_fwd']  # Fallback to eager

    # 3. Triton fast path
    print("\n[3] Triton fast path (eager)...")
    model.use_triton = True
    mean, std = time_forward(model, u, n_warmup=5, n_runs=20)
    tput = (batch * seqlen) / (mean / 1000)
    print(f"    {mean:7.2f} ± {std:5.2f} ms  |  {tput:>10,.0f} tokens/sec")
    results['triton_fwd'] = mean

    # 4. Triton + torch.compile
    print("\n[4] Triton fast path + torch.compile...")
    print("    Compiling model (this may take 30-60 seconds)...", end='', flush=True)
    model.use_triton = True
    try:
        with suppress_torch_compile_output():
            model_compiled = torch.compile(model, mode='reduce-overhead', fullgraph=False)
            mean, std = time_forward(model_compiled, u, n_warmup=10, n_runs=20)
        print(" done!")
        tput = (batch * seqlen) / (mean / 1000)
        print(f"    {mean:7.2f} ± {std:5.2f} ms  |  {tput:>10,.0f} tokens/sec")
        results['triton_compiled_fwd'] = mean
        del model_compiled
        torch.cuda.empty_cache()
    except Exception as e:
        print(" FAILED!")
        print(f"    Note: torch.compile incompatible with Triton kernels")
        print(f"    Skipping Triton+compile benchmarks")
        results['triton_compiled_fwd'] = results['triton_fwd']  # Fallback to eager

    # ========================================================================
    # BACKWARD PASS BENCHMARKS
    # ========================================================================
    print("\n" + "-"*70)
    print("FORWARD + BACKWARD PASS (training)")
    print("-"*70)

    model.train()

    # 5. Naive PyTorch backward
    print("\n[5] Naive PyTorch (eager)...")
    model.use_triton = False
    mean, std = time_backward(model, u, n_warmup=3, n_runs=10)
    tput = (batch * seqlen) / (mean / 1000)
    print(f"    {mean:7.2f} ± {std:5.2f} ms  |  {tput:>10,.0f} tokens/sec")
    results['naive_bwd'] = mean

    # 6. Naive + torch.compile backward
    print("\n[6] Naive PyTorch + torch.compile...")
    print("    Compiling model (this may take 30-60 seconds)...", end='', flush=True)
    model.use_triton = False
    try:
        with suppress_torch_compile_output():
            model_compiled = torch.compile(model, mode='reduce-overhead', fullgraph=False)
            mean, std = time_backward(model_compiled, u, n_warmup=5, n_runs=10)
        print(" done!")
        tput = (batch * seqlen) / (mean / 1000)
        print(f"    {mean:7.2f} ± {std:5.2f} ms  |  {tput:>10,.0f} tokens/sec")
        results['naive_compiled_bwd'] = mean
        del model_compiled
        torch.cuda.empty_cache()
    except Exception as e:
        print(" FAILED!")
        print(f"    torch.compile error: {type(e).__name__}")
        results['naive_compiled_bwd'] = results['naive_bwd']  # Fallback

    # 7. Triton backward
    print("\n[7] Triton fast path (eager)...")
    model.use_triton = True
    mean, std = time_backward(model, u, n_warmup=3, n_runs=10)
    tput = (batch * seqlen) / (mean / 1000)
    print(f"    {mean:7.2f} ± {std:5.2f} ms  |  {tput:>10,.0f} tokens/sec")
    results['triton_bwd'] = mean

    # 8. Triton + torch.compile backward
    print("\n[8] Triton fast path + torch.compile...")
    print("    Compiling model (this may take 30-60 seconds)...", end='', flush=True)
    model.use_triton = True
    try:
        with suppress_torch_compile_output():
            model_compiled = torch.compile(model, mode='reduce-overhead', fullgraph=False)
            mean, std = time_backward(model_compiled, u, n_warmup=5, n_runs=10)
        print(" done!")
        tput = (batch * seqlen) / (mean / 1000)
        print(f"    {mean:7.2f} ± {std:5.2f} ms  |  {tput:>10,.0f} tokens/sec")
        results['triton_compiled_bwd'] = mean
    except Exception as e:
        print(" FAILED!")
        print(f"    Note: torch.compile incompatible with Triton kernels")
        results['triton_compiled_bwd'] = results['triton_bwd']  # Fallback

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print("\nForward Pass Speedups:")
    baseline = results['naive_fwd']
    print(f"  Naive → Triton:                {baseline/results['triton_fwd']:6.2f}x")
    print(f"  Naive → Naive+compile:         {baseline/results['naive_compiled_fwd']:6.2f}x")
    print(f"  Naive → Triton+compile:        {baseline/results['triton_compiled_fwd']:6.2f}x")
    print(f"  Triton → Triton+compile:       {results['triton_fwd']/results['triton_compiled_fwd']:6.2f}x")

    print("\nBackward Pass Speedups:")
    baseline = results['naive_bwd']
    print(f"  Naive → Triton:                {baseline/results['triton_bwd']:6.2f}x")
    print(f"  Naive → Naive+compile:         {baseline/results['naive_compiled_bwd']:6.2f}x")
    print(f"  Naive → Triton+compile:        {baseline/results['triton_compiled_bwd']:6.2f}x")
    print(f"  Triton → Triton+compile:       {results['triton_bwd']/results['triton_compiled_bwd']:6.2f}x")

    print("\n" + "="*70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
