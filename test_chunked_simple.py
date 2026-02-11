#!/usr/bin/env python
"""Simple standalone test for Mamba2Chunked."""

import sys
import torch
import torch.nn.functional as F

# Add repo to path
sys.path.insert(0, '/home/tzeng/repos/chunk_mamba')

from mamba_ssm.modules.mamba2_chunked import Mamba2Chunked

def test_basic():
    """Basic smoke test."""
    print("Creating Mamba2Chunked model...")
    torch.manual_seed(42)

    model = Mamba2Chunked(
        d_model=128,
        d_state=16,
        d_conv=4,
        expand=2,
        headdim=32,
        ngroups=1,
        chunk_size=64,
        condition_x=True,
        condition_dt=True,
        condition_B=True,
        condition_C=True,
        use_triton=False,  # Start with naive path
        device="cuda",
        dtype=torch.float32,
    ).cuda()

    print(f"Model created. Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test input
    batch, seqlen, d_model = 2, 128, 128
    u = torch.randn(batch, seqlen, d_model, device="cuda", dtype=torch.float32)
    print(f"Input shape: {u.shape}")

    # Test naive path
    print("\n1. Testing naive path...")
    model.use_triton = False
    y_naive = model(u)
    print(f"   Output shape: {y_naive.shape}")
    print(f"   Output mean: {y_naive.mean().item():.6f}, std: {y_naive.std().item():.6f}")

    # Test Triton path
    print("\n2. Testing Triton path...")
    model.use_triton = True
    y_triton = model(u)
    print(f"   Output shape: {y_triton.shape}")
    print(f"   Output mean: {y_triton.mean().item():.6f}, std: {y_triton.std().item():.6f}")

    # Check agreement
    print("\n3. Checking naive vs Triton agreement...")
    diff = (y_naive - y_triton).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"   Max abs diff: {max_diff:.6e}")
    print(f"   Mean abs diff: {mean_diff:.6e}")

    if max_diff < 1e-3:
        print("   ✓ PASS: Naive and Triton paths agree!")
    else:
        print(f"   ✗ FAIL: Difference too large (max={max_diff:.6e})")
        return False

    # Test with R matrices activated
    print("\n4. Testing with non-zero R matrices...")
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.startswith("R_"):
                p.normal_(0, 0.01)

    model.use_triton = False
    y_naive2 = model(u)

    model.use_triton = True
    y_triton2 = model(u)

    diff2 = (y_naive2 - y_triton2).abs()
    max_diff2 = diff2.max().item()
    mean_diff2 = diff2.mean().item()
    print(f"   Max abs diff: {max_diff2:.6e}")
    print(f"   Mean abs diff: {mean_diff2:.6e}")

    if max_diff2 < 1e-3:
        print("   ✓ PASS: Naive and Triton paths agree with R matrices!")
    else:
        print(f"   ✗ FAIL: Difference too large (max={max_diff2:.6e})")
        return False

    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
    return True

if __name__ == "__main__":
    try:
        success = test_basic()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
