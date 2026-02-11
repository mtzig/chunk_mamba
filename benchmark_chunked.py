#!/usr/bin/env python
"""Timing benchmarks for Mamba2Chunked vs Mamba2Simple."""

import sys
import time
import torch
import torch.nn.functional as F
import os
from contextlib import contextmanager

# Add repo to path
sys.path.insert(0, '/home/tzeng/repos/chunk_mamba')

from mamba_ssm.modules.mamba2_chunked import Mamba2Chunked
from mamba_ssm.modules.mamba2_simple import Mamba2Simple


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


@contextmanager
def timing_context(name, warmup=False):
    """Context manager for timing with CUDA synchronization."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    yield
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if not warmup:
        print(f"  {name:40s}: {elapsed*1000:8.2f} ms")
    return elapsed


def benchmark_forward(model, u, n_warmup=5, n_runs=20, label=""):
    """Benchmark forward pass with warmup."""
    print(f"\n{label}")

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model(u)

    # Timed runs
    times = []
    for i in range(n_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            y = model(u)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    times = torch.tensor(times)
    mean_time = times.mean().item() * 1000  # ms
    std_time = times.std().item() * 1000
    min_time = times.min().item() * 1000

    batch, seqlen, _ = u.shape
    throughput = (batch * seqlen) / (mean_time / 1000)  # tokens/sec

    print(f"  Mean: {mean_time:7.2f} ms  ±{std_time:5.2f} ms")
    print(f"  Min:  {min_time:7.2f} ms")
    print(f"  Throughput: {throughput:,.0f} tokens/sec")

    return mean_time, throughput


def benchmark_backward(model, u, n_warmup=3, n_runs=10, label=""):
    """Benchmark forward + backward pass with warmup."""
    print(f"\n{label}")

    # Warmup
    for _ in range(n_warmup):
        u.requires_grad_(True)
        y = model(u)
        loss = y.sum()
        loss.backward()
        u.grad = None

    # Timed runs
    times = []
    for i in range(n_runs):
        u.requires_grad_(True)

        torch.cuda.synchronize()
        start = time.perf_counter()

        y = model(u)
        loss = y.sum()
        loss.backward()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        u.grad = None

    times = torch.tensor(times)
    mean_time = times.mean().item() * 1000  # ms
    std_time = times.std().item() * 1000
    min_time = times.min().item() * 1000

    batch, seqlen, _ = u.shape
    throughput = (batch * seqlen) / (mean_time / 1000)  # tokens/sec

    print(f"  Mean: {mean_time:7.2f} ms  ±{std_time:5.2f} ms")
    print(f"  Min:  {min_time:7.2f} ms")
    print(f"  Throughput: {throughput:,.0f} tokens/sec")

    return mean_time, throughput


def get_memory_stats():
    """Get current CUDA memory usage."""
    allocated = torch.cuda.memory_allocated() / 1024**2  # MB
    reserved = torch.cuda.memory_reserved() / 1024**2
    return allocated, reserved


def benchmark_suite(use_compile=False):
    """Run comprehensive benchmarks.

    Args:
        use_compile: If True, also benchmark torch.compile versions
    """
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float32

    configs = [
        # (batch, seqlen, d_model, chunk_size, label)
        (4, 1024, 768, 64, "Small (B=4, L=1024, D=128)"),
        (4, 2048, 768, 64, "Small (B=4, L=2048, D=128)"),
        # (4, 1024, 512, 256, "Medium (B=4, L=1024, D=512)"),
        # (2, 2048, 512, 256, "Large (B=2, L=2048, D=512)"),
        # (1, 4096, 512, 256, "Extra Large (B=1, L=4096, D=512)"),
    ]

    print("="*80)
    print("MAMBA2 CHUNKED BENCHMARK")
    print("="*80)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print("="*80)

    results = []

    for batch, seqlen, d_model, chunk_size, config_label in configs:
        print(f"\n{'='*80}")
        print(f"CONFIG: {config_label}")
        print(f"{'='*80}")

        # Create models
        print("\nCreating models...")

        model_chunked = Mamba2Chunked(
            d_model=d_model,
            d_state=64,
            d_conv=4,
            expand=2,
            headdim=64,
            ngroups=4,
            chunk_size=chunk_size,
            condition_x=True,
            condition_dt=True,
            condition_B=True,
            condition_C=True,
            use_triton=True,
            device=device,
            dtype=dtype,
        ).cuda().eval()

        model_simple = Mamba2Simple(
            d_model=d_model,
            d_state=64,
            d_conv=4,
            expand=2,
            headdim=64,
            ngroups=4,
            chunk_size=chunk_size,
            use_mem_eff_path=False,  # Unfused path for fair comparison
            device=device,
            dtype=dtype,
        ).cuda().eval()

        # Copy weights from chunked to simple for fair comparison
        with torch.no_grad():
            model_simple.in_proj.weight.copy_(model_chunked.in_proj.weight)
            model_simple.conv1d.weight.copy_(model_chunked.conv1d.weight)
            model_simple.conv1d.bias.copy_(model_chunked.conv1d.bias)
            model_simple.dt_bias.copy_(model_chunked.dt_bias)
            model_simple.A_log.copy_(model_chunked.A_log)
            model_simple.D.copy_(model_chunked.D)
            model_simple.norm.weight.copy_(model_chunked.norm.weight)
            model_simple.out_proj.weight.copy_(model_chunked.out_proj.weight)

        # Test input
        u = torch.randn(batch, seqlen, d_model, device=device, dtype=dtype)

        print(f"Input shape: {u.shape}")
        print(f"Total tokens: {batch * seqlen:,}")

        mem_before = get_memory_stats()
        print(f"Memory before: {mem_before[0]:.1f} MB allocated, {mem_before[1]:.1f} MB reserved")

        # ----------------------------------------------------------------
        # Forward-only benchmarks
        # ----------------------------------------------------------------
        print(f"\n{'-'*80}")
        print("FORWARD PASS BENCHMARKS")
        print(f"{'-'*80}")

        # Standard Mamba2 (baseline)
        mean_simple, tput_simple = benchmark_forward(
            model_simple, u, n_warmup=5, n_runs=20,
            label="[1] Mamba2Simple (baseline)"
        )

        # Baseline + torch.compile
        mean_simple_compiled = None
        tput_simple_compiled = None
        if use_compile:
            print(f"\n[2] Mamba2Simple + torch.compile")
            print("    Compiling... (may take 30-60 seconds)", end='', flush=True)
            try:
                with suppress_torch_compile_output():
                    model_simple_compiled = torch.compile(model_simple, mode='reduce-overhead', fullgraph=False)
                    mean_simple_compiled, tput_simple_compiled = benchmark_forward(
                        model_simple_compiled, u, n_warmup=10, n_runs=20,
                        label=""
                    )
                print(" done!")
                del model_simple_compiled
                torch.cuda.empty_cache()
            except Exception as e:
                print(" FAILED!")
                print(f"    torch.compile error: {type(e).__name__}")
                mean_simple_compiled = None

        # Chunked - Naive path
        model_chunked.use_triton = False
        label_idx = "3" if use_compile else "2"
        mean_naive, tput_naive = benchmark_forward(
            model_chunked, u, n_warmup=5, n_runs=20,
            label=f"[{label_idx}] Mamba2Chunked (naive PyTorch)"
        )

        # Chunked - Triton path (R=0, should match baseline)
        model_chunked.use_triton = True
        label_idx = "4" if use_compile else "3"
        mean_triton_r0, tput_triton_r0 = benchmark_forward(
            model_chunked, u, n_warmup=5, n_runs=20,
            label=f"[{label_idx}] Mamba2Chunked (Triton, R=0)"
        )

        # Chunked - Triton path with R matrices
        with torch.no_grad():
            for name, p in model_chunked.named_parameters():
                if name.startswith("R_"):
                    p.normal_(0, 0.01)

        label_idx = str(int(label_idx) + 1)
        mean_triton_r, tput_triton_r = benchmark_forward(
            model_chunked, u, n_warmup=5, n_runs=20,
            label=f"[{label_idx}] Mamba2Chunked (Triton, R≠0)"
        )

        # Triton + torch.compile
        mean_triton_compiled = None
        tput_triton_compiled = None
        if use_compile:
            label_idx = str(int(label_idx) + 1)
            print(f"\n[{label_idx}] Mamba2Chunked (Triton) + torch.compile (R≠0)")
            print("    Compiling... (may take 30-60 seconds)", end='', flush=True)
            try:
                with suppress_torch_compile_output():
                    model_compiled = torch.compile(model_chunked, mode='reduce-overhead', fullgraph=False)
                    mean_triton_compiled, tput_triton_compiled = benchmark_forward(
                        model_compiled, u, n_warmup=10, n_runs=20,
                        label=""
                    )
                print(" done!")
                del model_compiled
                torch.cuda.empty_cache()
            except Exception as e:
                print(" FAILED!")
                print(f"    Note: torch.compile incompatible with Triton kernels: {type(e).__name__}")
                mean_triton_compiled = None
                tput_triton_compiled = None

        # ----------------------------------------------------------------
        # Backward benchmarks
        # ----------------------------------------------------------------
        print(f"\n{'-'*80}")
        print("FORWARD + BACKWARD BENCHMARKS")
        print(f"{'-'*80}")

        mean_simple_bwd, tput_simple_bwd = benchmark_backward(
            model_simple, u.clone(), n_warmup=3, n_runs=10,
            label="[1] Mamba2Simple (baseline)"
        )

        model_chunked.use_triton = False
        mean_naive_bwd, tput_naive_bwd = benchmark_backward(
            model_chunked, u.clone(), n_warmup=3, n_runs=10,
            label="[2] Mamba2Chunked (naive PyTorch)"
        )

        model_chunked.use_triton = True
        # Zero out R matrices for fair comparison
        with torch.no_grad():
            for name, p in model_chunked.named_parameters():
                if name.startswith("R_"):
                    p.zero_()

        mean_triton_r0_bwd, tput_triton_r0_bwd = benchmark_backward(
            model_chunked, u.clone(), n_warmup=3, n_runs=10,
            label="[3] Mamba2Chunked (Triton, R=0)"
        )

        # Restore R matrices
        with torch.no_grad():
            for name, p in model_chunked.named_parameters():
                if name.startswith("R_"):
                    p.normal_(0, 0.01)

        mean_triton_bwd, tput_triton_bwd = benchmark_backward(
            model_chunked, u.clone(), n_warmup=3, n_runs=10,
            label="[4] Mamba2Chunked (Triton, R≠0)"
        )

        # Compiled version (if enabled)
        mean_triton_compiled_bwd = None
        tput_triton_compiled_bwd = None
        if use_compile and mean_triton_compiled is not None:  # Only try if forward succeeded
            print(f"\n[5] Mamba2Chunked (Triton+compile, R≠0)")
            print("    Compiling... (may take 30-60 seconds)", end='', flush=True)
            model_chunked.train()
            try:
                with suppress_torch_compile_output():
                    model_compiled = torch.compile(model_chunked, mode='reduce-overhead', fullgraph=False)
                    mean_triton_compiled_bwd, tput_triton_compiled_bwd = benchmark_backward(
                        model_compiled, u.clone(), n_warmup=5, n_runs=10,
                        label=""
                    )
                print(" done!")
                del model_compiled
                torch.cuda.empty_cache()
            except Exception as e:
                print(" FAILED!")
                print(f"    Note: torch.compile incompatible with Triton kernels: {type(e).__name__}")
                mean_triton_compiled_bwd = None
                tput_triton_compiled_bwd = None
            model_chunked.eval()

        # ----------------------------------------------------------------
        # Summary
        # ----------------------------------------------------------------
        print(f"\n{'-'*80}")
        print("SUMMARY")
        print(f"{'-'*80}")

        print(f"\nForward-only speedups vs baseline:")
        print(f"  Naive:          {mean_simple/mean_naive:.2f}x {'SLOWER' if mean_naive > mean_simple else 'FASTER'}")
        print(f"  Triton (R=0):   {mean_simple/mean_triton_r0:.2f}x {'SLOWER' if mean_triton_r0 > mean_simple else 'FASTER'}")
        print(f"  Triton (R≠0):   {mean_simple/mean_triton_r:.2f}x {'SLOWER' if mean_triton_r > mean_simple else 'FASTER'}")

        if use_compile:
            if mean_simple_compiled:
                print(f"  Simple+compile: {mean_simple/mean_simple_compiled:.2f}x {'SLOWER' if mean_simple_compiled > mean_simple else 'FASTER'}")
            if mean_triton_compiled:
                print(f"  Triton+compile: {mean_simple/mean_triton_compiled:.2f}x {'SLOWER' if mean_triton_compiled > mean_simple else 'FASTER'}")

        print(f"\nForward+Backward speedup vs baseline:")
        print(f"  Naive:          {mean_simple_bwd/mean_naive_bwd:.2f}x {'SLOWER' if mean_naive_bwd > mean_simple_bwd else 'FASTER'}")
        print(f"  Triton (R=0):   {mean_simple_bwd/mean_triton_r0_bwd:.2f}x {'SLOWER' if mean_triton_r0_bwd > mean_simple_bwd else 'FASTER'}")
        print(f"  Triton (R≠0):   {mean_simple_bwd/mean_triton_bwd:.2f}x {'SLOWER' if mean_triton_bwd > mean_simple_bwd else 'FASTER'}")
        if use_compile and mean_triton_compiled_bwd:
            print(f"  Triton+compile: {mean_simple_bwd/mean_triton_compiled_bwd:.2f}x {'SLOWER' if mean_triton_compiled_bwd > mean_simple_bwd else 'FASTER'}")

        overhead = ((mean_triton_r - mean_triton_r0) / mean_triton_r0) * 100
        print(f"\nChunk recurrence overhead: {overhead:.1f}%")

        if use_compile:
            print(f"\ntorch.compile speedups (eager → compiled):")
            if mean_simple_compiled:
                print(f"  Baseline fwd:   {mean_simple/mean_simple_compiled:.2f}x")
            if mean_triton_compiled:
                print(f"  Triton fwd:     {mean_triton_r/mean_triton_compiled:.2f}x")
            if mean_triton_compiled_bwd:
                print(f"  Triton bwd:     {mean_triton_bwd/mean_triton_compiled_bwd:.2f}x")

        mem_after = get_memory_stats()
        print(f"\nMemory after: {mem_after[0]:.1f} MB allocated, {mem_after[1]:.1f} MB reserved")
        print(f"Memory delta: {mem_after[0] - mem_before[0]:+.1f} MB")

        results.append({
            'config': config_label,
            'batch': batch,
            'seqlen': seqlen,
            'd_model': d_model,
            'simple_fwd': mean_simple,
            'simple_compiled_fwd': mean_simple_compiled,
            'naive_fwd': mean_naive,
            'triton_r0_fwd': mean_triton_r0,
            'triton_r_fwd': mean_triton_r,
            'triton_compiled_fwd': mean_triton_compiled,
            'simple_bwd': mean_simple_bwd,
            'naive_bwd': mean_naive_bwd,
            'triton_r0_bwd': mean_triton_r0_bwd,
            'triton_r_bwd': mean_triton_bwd,
            'triton_compiled_bwd': mean_triton_compiled_bwd,
        })

        # Cleanup
        del model_chunked, model_simple, u
        torch.cuda.empty_cache()

    # ----------------------------------------------------------------
    # Final summary table
    # ----------------------------------------------------------------
    print(f"\n\n{'='*80}")
    print("FINAL SUMMARY TABLE (Forward Pass)")
    print(f"{'='*80}\n")

    if use_compile:
        print(f"{'Config':<25} {'Baseline':<12} {'Triton(R≠0)':<12} {'T+compile':<12} {'Speedup':<10}")
        print(f"{'-'*80}")
        for r in results:
            speedup = r['simple_fwd'] / r['triton_r_fwd']
            if r['triton_compiled_fwd']:
                speedup_c = r['simple_fwd'] / r['triton_compiled_fwd']
                print(f"{r['config']:<25} {r['simple_fwd']:>8.2f} ms  {r['triton_r_fwd']:>8.2f} ms  "
                      f"{r['triton_compiled_fwd']:>8.2f} ms  {speedup:>5.2f}x/{speedup_c:>5.2f}x")
            else:
                print(f"{r['config']:<25} {r['simple_fwd']:>8.2f} ms  {r['triton_r_fwd']:>8.2f} ms  "
                      f"{'N/A':>8s}     {speedup:>7.2f}x")
    else:
        print(f"{'Config':<25} {'Baseline':<12} {'Triton(R≠0)':<12} {'Speedup':<10}")
        print(f"{'-'*80}")
        for r in results:
            speedup = r['simple_fwd'] / r['triton_r_fwd']
            print(f"{r['config']:<25} {r['simple_fwd']:>8.2f} ms  {r['triton_r_fwd']:>8.2f} ms  {speedup:>7.2f}x")

    print(f"\n{'='*80}")
    print("Benchmark complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark Mamba2Chunked")
    parser.add_argument('--compile', action='store_true',
                        help='Also benchmark torch.compile versions (takes longer)')
    args = parser.parse_args()

    try:
        benchmark_suite(use_compile=args.compile)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
