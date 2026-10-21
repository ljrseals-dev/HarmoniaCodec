from __future__ import annotations
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '.')
from harmonia_neural import (
    CausalConv1d, CausalTransposedConv1d, ResidualUnit,
    FocalModulation, AcousticEncoderNN, SubBandDecoderNN,
)

CONV_CONFIGS = [
    ("resid_k7_s1_d1",       7, 1, 1),
    ("resid_k7_s1_d2",       7, 1, 2),
    ("resid_k7_s1_d4",       7, 1, 4),
    ("identity_k1_s1_d1",    1, 1, 1),
    ("enc_s2_k5_d1",         5, 2, 1),
    ("enc_s2_k9_d1",         9, 2, 1),
    ("enc_s4_k9_d1",         9, 4, 1),
    ("enc_s5_k11_d1",       11, 5, 1),
    ("enc_s5_k11_d2",       11, 5, 2),
    ("enc_s3_k7_d1",         7, 3, 1),
    ("enc_s7_k15_d1",       15, 7, 1),
]
CHUNK_SIZES = [8, 16, 32, 64, 128]
TOTAL_LENGTHS = [256, 512]


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
    y_chunked = torch.cat(outs, dim=-1)
    if y_single.shape[-1] != y_chunked.shape[-1]:
        return (False, float("inf"),
                f"length mismatch: single={y_single.shape[-1]} chunked={y_chunked.shape[-1]}")
    if y_single.numel() == 0:
        return True, 0.0, "empty output on both paths"
    diff = (y_single - y_chunked).abs().max().item()
    ok = diff <= atol
    detail = f"max abs diff {diff:.3e}" + ("" if ok else f" exceeds atol {atol:.1e}")
    return ok, diff, detail


def length_preserved(module, in_channels, L):
    module.reset()
    x = torch.randn(1, in_channels, L)
    with torch.no_grad():
        y = module(x)
    return y.shape[-1] == L


def test_causal_conv_streaming():
    print("\n[CausalConv1d] streaming equivalence")
    failures = []
    n = 0
    for name, k, s, d in CONV_CONFIGS:
        for chunk in CHUNK_SIZES:
            for total in TOTAL_LENGTHS:
                n += 1
                m = CausalConv1d(4, 4, k, stride=s, dilation=d)
                ok, diff, msg = streaming_matches_single_pass(
                    m, total_len=total, chunk_size=chunk, in_channels=4,
                    seed=abs(hash((name, chunk, total))) % (2**31))
                if not ok:
                    failures.append((name, chunk, total, msg))
                    print(f"  FAIL {name:<20} chunk={chunk:>3} total={total:>4}  {msg}")
    print(f"  {n - len(failures)}/{n} configs pass")
    return failures


def test_causal_conv_length_preserved():
    print("\n[CausalConv1d] length preservation at stride=1")
    failures = []
    for name, k, s, d in CONV_CONFIGS:
        if s != 1:
            continue
        m = CausalConv1d(4, 4, k, stride=s, dilation=d)
        for L in (16, 32, 64, 128):
            if not length_preserved(m, 4, L):
                failures.append((name, L))
                print(f"  FAIL {name} L={L}: length not preserved")
    print(f"  {'all pass' if not failures else f'{len(failures)} failures'}")
    return failures


def test_causal_conv_kernel_one():
    print("\n[CausalConv1d] kernel=1 receptive=0")
    failures = []
    m = CausalConv1d(4, 4, 1, stride=1)
    for chunk in (8, 16, 32):
        ok, diff, msg = streaming_matches_single_pass(
            m, total_len=128, chunk_size=chunk, in_channels=4, seed=chunk)
        if not ok:
            failures.append((chunk, msg))
            print(f"  FAIL chunk={chunk}: {msg}")
    print(f"  {'pass' if not failures else 'FAIL'}")
    return failures


def test_causal_transposed_conv_streaming():
    print("\n[CausalTransposedConv1d] streaming equivalence")
    failures = []
    n = 0
    for name, k, s, d in CONV_CONFIGS:
        if k == 1:
            continue
        for chunk in CHUNK_SIZES:
            for total in TOTAL_LENGTHS:
                n += 1
                m = CausalTransposedConv1d(4, 4, k, stride=s, dilation=d)
                ok, diff, msg = streaming_matches_single_pass(
                    m, total_len=total, chunk_size=chunk, in_channels=4,
                    seed=abs(hash((name, chunk, total, "tr"))) % (2**31),
                    atol=1e-4)
                if not ok:
                    failures.append((name, chunk, total, msg))
                    print(f"  FAIL {name:<20} chunk={chunk:>3} total={total:>4}  {msg}")
    print(f"  {n - len(failures)}/{n} configs pass")
    return failures


def test_residual_unit_streaming():
    print("\n[ResidualUnit] streaming equivalence")
    failures = []
    for dilation in (1, 2, 4):
        m = ResidualUnit(8, dilation=dilation, causal=True)
        for chunk in (16, 32, 64, 128):
            ok, diff, msg = streaming_matches_single_pass(
                m, total_len=512, chunk_size=chunk, in_channels=8,
                seed=dilation * 100 + chunk)
            if not ok:
                failures.append((dilation, chunk, msg))
                print(f"  FAIL dilation={dilation} chunk={chunk}: {msg}")
    print(f"  {'all pass' if not failures else f'{len(failures)} failures'}")
    return failures


def test_focal_modulation_streaming():
    print("\n[FocalModulation] streaming equivalence")
    failures = []
    for chunk in (8, 16, 32, 64):
        m = FocalModulation(8, kernel=7, causal=True)
        ok, diff, msg = streaming_matches_single_pass(
            m, total_len=256, chunk_size=chunk, in_channels=8, seed=chunk)
        if not ok:
            failures.append((chunk, msg))
            print(f"  FAIL chunk={chunk}: {msg}")
    print(f"  {'all pass' if not failures else f'{len(failures)} failures'}")
    return failures


def test_encoder_streaming():
    print("\n[AcousticEncoderNN] streaming equivalence")
    failures = []
    for chunk in (64, 128, 256):
        m = AcousticEncoderNN(dim=16, latent=8, strides=(2, 4, 5, 5), causal=True)
        ok, diff, msg = streaming_matches_single_pass(
            m, total_len=1024, chunk_size=chunk, in_channels=1, seed=chunk, atol=1e-4)
        if not ok:
            failures.append((chunk, msg))
            print(f"  FAIL chunk={chunk}: {msg}")
    print(f"  {'all pass' if not failures else f'{len(failures)} failures'}")
    return failures


def test_decoder_streaming():
    print("\n[SubBandDecoderNN] streaming equivalence")
    failures = []
    for chunk in (16, 32, 64):
        m = SubBandDecoderNN(latent=8, dim=16, strides=(5, 5, 4, 2), causal=True)
        ok, diff, msg = streaming_matches_single_pass(
            m, total_len=256, chunk_size=chunk, in_channels=8, seed=chunk, atol=1e-4)
        if not ok:
            failures.append((chunk, msg))
            print(f"  FAIL chunk={chunk}: {msg}")
    print(f"  {'all pass' if not failures else f'{len(failures)} failures'}")
    return failures


if __name__ == "__main__":
    total_failures = 0
    for t in [test_causal_conv_streaming, test_causal_conv_length_preserved,
              test_causal_conv_kernel_one, test_causal_transposed_conv_streaming,
              test_residual_unit_streaming, test_focal_modulation_streaming,
              test_encoder_streaming, test_decoder_streaming]:
        f = t()
        total_failures += len(f)
    print("\n" + "="*72)
    print("ALL PASS" if total_failures == 0 else f"{total_failures} TOTAL FAILURES")
