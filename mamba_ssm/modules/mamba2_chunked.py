# Copyright (c) 2024, Tri Dao, Albert Gu.
# Chunkwise Generalized Mamba2 implementation.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None

try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
except ImportError:
    RMSNormGated = None

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined


class Mamba2Chunked(nn.Module):
    """Chunkwise Generalized Mamba2.

    Splits the sequence into contiguous chunks of length ``chunk_size``.  For
    each chunk the gates (dt, x, B, C) receive an additive offset that is a
    learned linear function of the previous chunk's output summary
    ``s_{k-1} = sum_{t in chunk k-1} y_t``.

    Two forward paths are provided:
      * ``forward_naive``  – pure-PyTorch per-timestep recurrence (Stage 1).
      * ``forward_triton`` – calls the existing Triton ``mamba_chunk_scan_combined``
        kernel once per chunk with modified projections (Stage 2).

    The default ``forward`` dispatches based on ``self.use_triton``.
    """

    def __init__(
        self,
        d_model,
        d_state=64,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=128,
        ngroups=1,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        activation="swish",
        bias=False,
        conv_bias=True,
        chunk_size=256,
        use_mem_eff_path=False,
        layer_idx=None,
        # Chunk recurrence options
        condition_x=True,
        condition_dt=True,
        condition_B=True,
        condition_C=True,
        use_triton=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        self.chunk_size = chunk_size
        self.use_triton = use_triton
        self.layer_idx = layer_idx

        self.condition_x = condition_x
        self.condition_dt = condition_dt
        self.condition_B = condition_B
        self.condition_C = condition_C

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)

        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        if self.learnable_init_states:
            self.init_states = nn.Parameter(
                torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs)
            )
            self.init_states._no_weight_decay = True

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # A parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        # Extra normalization layer right before output projection
        assert RMSNormGated is not None
        self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False, **factory_kwargs)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        # ------------------------------------------------------------------
        # Chunk-recurrence R matrices
        # Each maps s_prev in R^{d_inner} to the corresponding offset.
        # Zero-initialized so the model starts as standard Mamba2.
        # ------------------------------------------------------------------
        if self.condition_x:
            self.R_x = nn.Linear(self.d_inner, self.d_inner, bias=False, **factory_kwargs)
            nn.init.zeros_(self.R_x.weight)
        if self.condition_dt:
            self.R_dt = nn.Linear(self.d_inner, self.nheads, bias=False, **factory_kwargs)
            nn.init.zeros_(self.R_dt.weight)
        if self.condition_B:
            self.R_B = nn.Linear(self.d_inner, self.ngroups * self.d_state, bias=False, **factory_kwargs)
            nn.init.zeros_(self.R_B.weight)
        if self.condition_C:
            self.R_C = nn.Linear(self.d_inner, self.ngroups * self.d_state, bias=False, **factory_kwargs)
            nn.init.zeros_(self.R_C.weight)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _base_projections(self, u):
        """Compute the base projections for all tokens.

        Returns (z, x, B, C, dt_raw) where x/B/C have already been through
        conv1d + SiLU and dt_raw is *before* softplus.
        """
        batch, seqlen, _ = u.shape

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        z, xBC, dt_raw = torch.split(
            zxbcdt,
            [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1,
        )

        # Conv1d + SiLU on xBC
        assert self.activation in ["silu", "swish"]
        if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
            xBC = self.act(self.conv1d(xBC.transpose(1, 2)).transpose(1, 2))
            xBC = xBC[:, :seqlen, :]
        else:
            xBC = causal_conv1d_fn(
                x=xBC.transpose(1, 2),
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
            ).transpose(1, 2)

        x, B, C = torch.split(
            xBC,
            [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1,
        )
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        B = rearrange(B, "b l (g n) -> b l g n", g=self.ngroups)
        C = rearrange(C, "b l (g n) -> b l g n", g=self.ngroups)

        return z, x, B, C, dt_raw

    def _apply_chunk_offsets(self, x_k, B_k, C_k, dt_k, s_prev):
        """Add chunk-recurrent offsets derived from *s_prev* to the chunk's
        projections.  Following the "chunkwise convolutional version" from the
        spec, offsets are linear functions of s_prev added *after* the base
        nonlinearities (SiLU / conv1d) for x/B/C, and *before* softplus for dt.
        """
        if self.condition_x:
            x_offset = self.R_x(s_prev)  # (batch, d_inner)
            x_k = x_k + rearrange(x_offset, "b (h p) -> b 1 h p", p=self.headdim)
        if self.condition_dt:
            dt_offset = self.R_dt(s_prev)  # (batch, nheads)
            dt_k = dt_k + dt_offset.unsqueeze(1)
        if self.condition_B:
            B_offset = self.R_B(s_prev)  # (batch, ngroups * d_state)
            B_k = B_k + rearrange(B_offset, "b (g n) -> b 1 g n", g=self.ngroups)
        if self.condition_C:
            C_offset = self.R_C(s_prev)  # (batch, ngroups * d_state)
            C_k = C_k + rearrange(C_offset, "b (g n) -> b 1 g n", g=self.ngroups)
        return x_k, B_k, C_k, dt_k

    # ------------------------------------------------------------------
    # Stage 1: Naive PyTorch reference
    # ------------------------------------------------------------------

    def forward_naive(self, u, seq_idx=None):
        """Pure PyTorch per-timestep recurrence.  Slow but useful as a
        correctness reference."""
        batch, seqlen, _ = u.shape
        z, x, B, C, dt_raw = self._base_projections(u)

        A = -torch.exp(self.A_log.float())  # (nheads,)
        ngroups_ratio = self.nheads // self.ngroups

        nchunks = math.ceil(seqlen / self.chunk_size)
        s_prev = x.new_zeros(batch, self.d_inner)
        # SSM state – always fp32 for stability
        c = x.new_zeros(batch, self.nheads, self.headdim, self.d_state, dtype=torch.float32)

        if self.learnable_init_states:
            c = c + repeat(self.init_states, "h p n -> b h p n", b=batch)

        all_outputs = []
        for k in range(nchunks):
            start = k * self.chunk_size
            end = min((k + 1) * self.chunk_size, seqlen)
            chunk_len = end - start

            # Slice this chunk's base projections
            x_k = x[:, start:end]
            B_k = B[:, start:end]
            C_k = C[:, start:end]
            dt_k = dt_raw[:, start:end]

            # Apply chunk-recurrent offsets from s_prev
            x_k, B_k, C_k, dt_k = self._apply_chunk_offsets(x_k, B_k, C_k, dt_k, s_prev)

            # softplus(dt_raw + dt_bias)
            dt_k = F.softplus(dt_k + self.dt_bias)  # (batch, chunk_len, nheads)

            # dt_limit clamping
            if self.dt_limit != (0.0, float("inf")):
                dt_k = dt_k.clamp(min=self.dt_limit[0], max=self.dt_limit[1])

            # --- naive recurrence over timesteps in the chunk ---
            chunk_outputs = []
            for t in range(chunk_len):
                dt_t = dt_k[:, t]  # (batch, nheads)
                decay = torch.exp(A * dt_t)  # (batch, nheads)

                # Expand B/C from ngroups to nheads
                B_t = B_k[:, t].repeat_interleave(ngroups_ratio, dim=1)  # (batch, nheads, d_state)
                C_t = C_k[:, t].repeat_interleave(ngroups_ratio, dim=1)  # (batch, nheads, d_state)

                x_t = x_k[:, t]  # (batch, nheads, headdim)

                # State update:  c = decay * c + dt * x ⊗ B
                c = (
                    decay[:, :, None, None] * c
                    + dt_t[:, :, None, None] * x_t[:, :, :, None] * B_t[:, :, None, :]
                )

                # Output: y_t = c @ C_t  (contract d_state)
                y_t = torch.einsum("bhpn,bhn->bhp", c.to(x_t.dtype), C_t)

                # D skip connection
                y_t = y_t + self.D[None, :, None] * x_t

                chunk_outputs.append(y_t)

            # (batch, chunk_len, nheads, headdim)
            chunk_out = torch.stack(chunk_outputs, dim=1)
            all_outputs.append(chunk_out)

            # Update chunk summary:  s_k = sum_t y_t  (in model dim d_inner)
            s_prev = rearrange(chunk_out.sum(dim=1), "b h p -> b (h p)")

        y = torch.cat(all_outputs, dim=1)  # (batch, seqlen, nheads, headdim)
        y = rearrange(y, "b l h p -> b l (h p)")

        # Gating + norm + output projection
        y = self.norm(y, z)
        return self.out_proj(y)

    # ------------------------------------------------------------------
    # Stage 2: Triton fast path
    # ------------------------------------------------------------------

    def forward_triton(self, u, seq_idx=None):
        """Fast path using the repo's existing Triton SSD/chunk-scan kernels.

        Iterates over outer chunks on the host.  For each chunk:
          1. Compute offsets from s_prev and broadcast-add to that chunk's
             base projections.
          2. Call ``mamba_chunk_scan_combined`` (with ``initial_states``) to
             process the chunk.
          3. Sum-pool the chunk outputs to update s_prev.
        """
        batch, seqlen, _ = u.shape
        z, x, B, C, dt_raw = self._base_projections(u)

        A = -torch.exp(self.A_log.float())  # (nheads,)
        dt_limit_kwargs = (
            {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        )

        nchunks = math.ceil(seqlen / self.chunk_size)
        s_prev = x.new_zeros(batch, self.d_inner)

        initial_states = None
        if self.learnable_init_states:
            initial_states = repeat(self.init_states, "h p n -> b h p n", b=batch)

        all_outputs = []
        for k in range(nchunks):
            start = k * self.chunk_size
            end = min((k + 1) * self.chunk_size, seqlen)
            chunk_len = end - start

            # Slice (contiguous copies for Triton)
            x_k = x[:, start:end].contiguous()
            B_k = B[:, start:end].contiguous()
            C_k = C[:, start:end].contiguous()
            dt_k = dt_raw[:, start:end].contiguous()

            # Apply chunk-recurrent offsets
            x_k, B_k, C_k, dt_k = self._apply_chunk_offsets(x_k, B_k, C_k, dt_k, s_prev)

            # softplus(dt + dt_bias)
            dt_k = F.softplus(dt_k + self.dt_bias)  # (batch, chunk_len, nheads)

            if self.dt_limit != (0.0, float("inf")):
                dt_k = dt_k.clamp(min=self.dt_limit[0], max=self.dt_limit[1])

            # Use the Triton chunk-scan kernel to process this chunk
            result = mamba_chunk_scan_combined(
                x_k,
                dt_k,
                A,
                B_k,
                C_k,
                chunk_size=chunk_len,
                D=self.D,
                z=None,
                initial_states=initial_states,
                return_final_states=True,
                **dt_limit_kwargs,
            )
            y_k, final_states_k = result

            all_outputs.append(y_k)
            initial_states = final_states_k

            # Update chunk summary
            s_prev = rearrange(y_k.sum(dim=1), "b h p -> b (h p)")

        y = torch.cat(all_outputs, dim=1)  # (batch, seqlen, nheads, headdim)
        y = rearrange(y, "b l h p -> b l (h p)")

        # Gating + norm + output projection
        y = self.norm(y, z)
        return self.out_proj(y)

    # ------------------------------------------------------------------
    # Default forward
    # ------------------------------------------------------------------

    def forward(self, u, seq_idx=None):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        if self.use_triton:
            return self.forward_triton(u, seq_idx=seq_idx)
        else:
            return self.forward_naive(u, seq_idx=seq_idx)
