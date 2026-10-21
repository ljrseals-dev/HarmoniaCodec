#!/usr/bin/env python3
from __future__ import annotations
import math, sys
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1dUnderTest(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride=1, dilation=1, groups=1):
        super().__init__()
        self.kernel, self.stride, self.dilation = kernel, stride, dilation
        self.pad = (kernel - 1) * dilation
        self.keep = self.pad - self.stride + 1
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride=stride,
                              dilation=dilation, groups=groups)
        self._state: Optional[torch.Tensor] = None

    def reset(self): self._state = None

    def forward(self, x):
        if self._state is not None:
            x = torch.cat([self._state, x], dim=-1)
        x = F.pad(x, (self.pad, 0))
        y = self.conv(x)
        if self.keep > 0:
            self._state = x[..., -self.keep:].detach()
        return y


class CausalConv1dFixed(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride=1, dilation=1, groups=1):
        super().__init__()
        self.kernel, self.stride, self.dilation = kernel, stride, dilation
        self.pad = (kernel - 1) * dilation
        self.keep = self.pad - self.stride + 1
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride=stride,
                              dilation=dilation, groups=groups)
        self._state: Optional[torch.Tensor] = None

    def reset(self): self._state = None

    def forward(self, x):
        had_state = self._state is not None
        if had_state:
            x = torch.cat([self._state, x], dim=-1)
        x = F.pad(x, (self.pad, 0))
        y = self.conv(x)
        if had_state and self.keep > 0:
            y = y[..., self.keep:]
        if self.keep > 0:
            self._state = x[..., -self.keep:].detach()
        return y


class ResidualUnitUnderTest(nn.Module):
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.c1 = CausalConv1dUnderTest(dim, dim, 7, dilation=dilation)
        self.c2 = CausalConv1dUnderTest(dim, dim, 1)
        self.act = nn.ELU()
    def reset(self): self.c1.reset(); self.c2.reset()
    def forward(self, x):
        return x + self.c2(self.act(self.c1(self.act(x))))


class ResidualUnitFixed(nn.Module):
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.c1 = CausalConv1dFixed(dim, dim, 7, dilation=dilation)
        self.c2 = CausalConv1dFixed(dim, dim, 1)
        self.act = nn.ELU()
    def reset(self): self.c1.reset(); self.c2.reset()
    def forward(self, x):
        return x + self.c2(self.act(self.c1(self.act(x))))


def streaming_matches_single_pass(module, total_len, chunk_size, in_channels,
                                   *, seed=0, atol=1e-5):
    torch.manual_seed(seed)
    x = torch.randn(1, in_channels, total_len)
    module.reset()
    with torch.no_grad():
        y_single = module(x).clone()
    module.reset()
    outs = []
    for start in range(0, total_len, chunk_size):
        chunk = x[..., start:start + chunk_size]
        with torch.no_grad():
            outs.append(module(chunk).clone())
    y_stream = torch.cat(outs, dim=-1)
    if y_single.shape[-1] != y_stream.shape[-1]:
        return (False, float("inf"),
                f"length mismatch: single-pass={y_single.shape[-1]} streamed={y_stream.shape[-1]}")
    diff = (y_single - y_stream).abs().max().item()
    ok = diff <= atol
    msg = "identical to within tolerance" if ok else f"max abs diff {diff:.3e} exceeds atol {atol:.1e}"
    return ok, diff, msg


CASES = [(3,1,1),(5,1,1),(7,1,1),(7,1,2),(11,1,1),(3,2,1),(5,2,1),(7,2,1),(11,5,1)]


def test_unfixed_fails_as_expected():
    print("\n[unfixed] confirming the bug is real: each case should FAIL")
    all_failed = True
    for kernel, stride, dilation in CASES:
        module = CausalConv1dUnderTest(4, 4, kernel, stride=stride, dilation=dilation)
        ok, diff, msg = streaming_matches_single_pass(
            module, total_len=64, chunk_size=16, in_channels=4, seed=kernel)
        status = "PASS" if ok else "FAIL"
        print(f"  kernel={kernel:>2} stride={stride} dilation={dilation}: {status}  ({msg})")
        if ok:
            all_failed = False
    assert not all_failed, "unfixed conv unexpectedly passed everywhere -- premise invalid"
    print("  premise confirmed: unfixed conv fails streaming equivalence")


def test_unfixed_residual_unit_crashes():
    print("\n[unfixed] confirming ResidualUnit crashes on second chunk")
    module = ResidualUnitUnderTest(dim=8, dilation=1)
    x1 = torch.randn(1, 8, 32)
    x2 = torch.randn(1, 8, 32)
    module.reset()
    with torch.no_grad():
        module(x1)
    crashed = False
    try:
        with torch.no_grad():
            module(x2)
    except RuntimeError as e:
        crashed = True
        print(f"  crash on second chunk (expected): {type(e).__name__}: {e}")
    assert crashed, "unfixed ResidualUnit did NOT crash on second chunk"


def test_fixed_passes_across_configs():
    print("\n[fixed] streaming equivalence across configs")
    failures = []
    for kernel, stride, dilation in CASES:
        for chunk_size in (8, 16, 32):
            module = CausalConv1dFixed(4, 4, kernel, stride=stride, dilation=dilation)
            ok, diff, msg = streaming_matches_single_pass(
                module, total_len=128, chunk_size=chunk_size, in_channels=4,
                seed=kernel * 100 + chunk_size)
            status = "PASS" if ok else "FAIL"
            print(f"  kernel={kernel:>2} stride={stride} dilation={dilation} chunk={chunk_size:>3}: {status}  ({msg})")
            if not ok:
                failures.append((kernel, stride, dilation, chunk_size, msg))
    assert not failures, f"{len(failures)} configs failed: {failures}"


def test_fixed_residual_unit_passes_streaming():
    print("\n[fixed] ResidualUnit streaming equivalence")
    for chunk_size in (16, 32, 64):
        module = ResidualUnitFixed(dim=8, dilation=1)
        ok, diff, msg = streaming_matches_single_pass(
            module, total_len=256, chunk_size=chunk_size, in_channels=8, seed=chunk_size)
        status = "PASS" if ok else "FAIL"
        print(f"  chunk={chunk_size:>3}: {status}  ({msg})")
        assert ok, f"ResidualUnit streaming failed at chunk={chunk_size}: {msg}"


def test_reset_clears_state():
    print("\n[fixed] reset() clears state")
    module = CausalConv1dFixed(4, 4, 7)
    x = torch.randn(1, 4, 32)
    module.reset()
    with torch.no_grad():
        a1 = module(x); _ = module(x)
    module.reset()
    with torch.no_grad():
        b1 = module(x)
    assert torch.allclose(a1, b1, atol=1e-6), "reset() did not clear state"
    print("  reset() ok")


def main():
    print("=" * 72)
    print("causal streaming equivalence -- property test")
    print("=" * 72)
    failures = 0
    for t in [test_unfixed_fails_as_expected, test_unfixed_residual_unit_crashes,
              test_fixed_passes_across_configs, test_fixed_residual_unit_passes_streaming,
              test_reset_clears_state]:
        try:
            t()
        except AssertionError as e:
            print(f"\n  ASSERTION FAILED in {t.__name__}: {e}")
            failures += 1
    print("\n" + "=" * 72)
    print("ALL PASS." if failures == 0 else f"{failures} FAILURE(S).")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
