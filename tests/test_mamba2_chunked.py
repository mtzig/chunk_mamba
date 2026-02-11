"""Tests for Mamba2Chunked: verify naive ↔ Triton numerical agreement."""

import math
import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from mamba_ssm.modules.mamba2_chunked import Mamba2Chunked


def _make_model(d_model=128, chunk_size=64, dtype=torch.float32, **kwargs):
    model = Mamba2Chunked(
        d_model=d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        headdim=32,
        ngroups=1,
        chunk_size=chunk_size,
        condition_x=True,
        condition_dt=True,
        condition_B=True,
        condition_C=True,
        device="cuda",
        dtype=dtype,
        **kwargs,
    )
    return model.cuda()


# ------------------------------------------------------------------
# 1. Basic smoke test
# ------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32])
def test_smoke(dtype):
    """Module runs without error in both modes."""
    torch.manual_seed(42)
    model = _make_model(dtype=dtype)
    u = torch.randn(2, 256, 128, device="cuda", dtype=dtype)

    model.use_triton = False
    y_naive = model(u)
    assert y_naive.shape == u.shape

    model.use_triton = True
    y_triton = model(u)
    assert y_triton.shape == u.shape


# ------------------------------------------------------------------
# 2. Naive == Triton (fp32, strict)
# ------------------------------------------------------------------

@pytest.mark.parametrize("seqlen", [64, 128, 256, 300])
def test_naive_triton_agreement_fp32(seqlen):
    """Naive and Triton paths should agree in fp32."""
    torch.manual_seed(0)
    model = _make_model(d_model=128, chunk_size=64, dtype=torch.float32)

    # Give the R matrices some non-trivial values
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.startswith("R_"):
                p.normal_(0, 0.01)

    u = torch.randn(2, seqlen, 128, device="cuda", dtype=torch.float32)

    model.use_triton = False
    y_naive = model(u)

    model.use_triton = True
    y_triton = model(u)

    torch.testing.assert_close(y_naive, y_triton, atol=1e-4, rtol=1e-4)


# ------------------------------------------------------------------
# 3. Naive == Triton (bf16, looser tolerance)
# ------------------------------------------------------------------

@pytest.mark.parametrize("seqlen", [128, 256])
def test_naive_triton_agreement_bf16(seqlen):
    """Naive and Triton paths should agree in bf16 within loose tolerance."""
    torch.manual_seed(1)
    model = _make_model(d_model=128, chunk_size=64, dtype=torch.bfloat16)

    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.startswith("R_"):
                p.normal_(0, 0.01)

    u = torch.randn(2, seqlen, 128, device="cuda", dtype=torch.bfloat16)

    model.use_triton = False
    y_naive = model(u)

    model.use_triton = True
    y_triton = model(u)

    torch.testing.assert_close(y_naive, y_triton, atol=5e-2, rtol=5e-2)


# ------------------------------------------------------------------
# 4. Zero R matrices → standard Mamba2 equivalence
# ------------------------------------------------------------------

def test_zero_R_matches_mamba2():
    """With R=0, Mamba2Chunked should match standard Mamba2Simple."""
    from mamba_ssm.modules.mamba2_simple import Mamba2Simple

    torch.manual_seed(42)
    d_model, chunk_size = 128, 64
    dtype = torch.float32

    chunked = _make_model(d_model=d_model, chunk_size=chunk_size, dtype=dtype)

    # Build a Mamba2Simple with identical base weights
    simple = Mamba2Simple(
        d_model=d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        headdim=32,
        ngroups=1,
        chunk_size=chunk_size,
        use_mem_eff_path=False,
        device="cuda",
        dtype=dtype,
    )

    # Copy base weights from chunked → simple
    with torch.no_grad():
        simple.in_proj.weight.copy_(chunked.in_proj.weight)
        if simple.in_proj.bias is not None:
            simple.in_proj.bias.copy_(chunked.in_proj.bias)
        simple.conv1d.weight.copy_(chunked.conv1d.weight)
        simple.conv1d.bias.copy_(chunked.conv1d.bias)
        simple.dt_bias.copy_(chunked.dt_bias)
        simple.A_log.copy_(chunked.A_log)
        simple.D.copy_(chunked.D)
        simple.norm.weight.copy_(chunked.norm.weight)
        simple.out_proj.weight.copy_(chunked.out_proj.weight)
        if simple.out_proj.bias is not None:
            simple.out_proj.bias.copy_(chunked.out_proj.bias)

    # Ensure R matrices are zero (they should be by default)
    with torch.no_grad():
        for name, p in chunked.named_parameters():
            if name.startswith("R_"):
                assert (p == 0).all(), f"{name} should be zero-initialized"

    u = torch.randn(2, 128, d_model, device="cuda", dtype=dtype)

    # Chunked (Triton path) with R=0 should match standard Mamba2
    chunked.use_triton = True
    y_chunked = chunked(u)
    y_simple = simple(u)

    torch.testing.assert_close(y_chunked, y_simple, atol=1e-4, rtol=1e-4)


# ------------------------------------------------------------------
# 5. Gradient flows through both paths
# ------------------------------------------------------------------

@pytest.mark.parametrize("use_triton", [False, True])
def test_gradient_flow(use_triton):
    """Verify gradients flow back to all parameters."""
    torch.manual_seed(7)
    model = _make_model(dtype=torch.float32)
    model.use_triton = use_triton

    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.startswith("R_"):
                p.normal_(0, 0.01)

    u = torch.randn(2, 128, 128, device="cuda", dtype=torch.float32, requires_grad=True)
    y = model(u)
    loss = y.sum()
    loss.backward()

    # Check that gradients exist for key parameters
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(p.grad).any(), f"NaN gradient for {name}"


# ------------------------------------------------------------------
# 6. Non-divisible sequence length
# ------------------------------------------------------------------

def test_non_divisible_seqlen():
    """Sequence length not divisible by chunk_size should still work."""
    torch.manual_seed(3)
    model = _make_model(d_model=128, chunk_size=64, dtype=torch.float32)

    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.startswith("R_"):
                p.normal_(0, 0.01)

    u = torch.randn(1, 100, 128, device="cuda", dtype=torch.float32)

    model.use_triton = False
    y_naive = model(u)

    model.use_triton = True
    y_triton = model(u)

    torch.testing.assert_close(y_naive, y_triton, atol=1e-4, rtol=1e-4)


# ------------------------------------------------------------------
# 7. Selective conditioning
# ------------------------------------------------------------------

@pytest.mark.parametrize(
    "cond",
    [
        dict(condition_x=True, condition_dt=False, condition_B=False, condition_C=False),
        dict(condition_x=False, condition_dt=True, condition_B=False, condition_C=False),
        dict(condition_x=False, condition_dt=False, condition_B=True, condition_C=True),
    ],
)
def test_selective_conditioning(cond):
    """Only some R matrices enabled; naive and Triton should still agree."""
    torch.manual_seed(5)
    model = Mamba2Chunked(
        d_model=128,
        d_state=16,
        d_conv=4,
        expand=2,
        headdim=32,
        ngroups=1,
        chunk_size=64,
        device="cuda",
        dtype=torch.float32,
        **cond,
    ).cuda()

    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.startswith("R_"):
                p.normal_(0, 0.01)

    u = torch.randn(2, 128, 128, device="cuda", dtype=torch.float32)

    model.use_triton = False
    y_naive = model(u)

    model.use_triton = True
    y_triton = model(u)

    torch.testing.assert_close(y_naive, y_triton, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
