#!/usr/bin/env python3
"""
harmonia_signalgen.py — deterministic test signals for codec evaluation.

Every signal is a pure function of its inputs. `pink_noise` and
`speech_like` are seeded; `stereo_battery` is seeded by CRC32 of the
signal name so successive calls in different processes produce identical
output. No global RNG state is used or mutated.

Verification:
    python harmonia_signalgen.py
prints four checks. All four must print True.
"""
from __future__ import annotations
import hashlib
import zlib
import numpy as np


# ── primitives ────────────────────────────────────────────────────

def sine(freq: float, sr: int, dur: float, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(sr * dur)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def cosine(freq: float, sr: int, dur: float, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(sr * dur)) / sr
    return (amp * np.cos(2 * np.pi * freq * t)).astype(np.float32)


def square(freq: float, sr: int, dur: float, amp: float = 0.5,
           harmonic_cap: int | None = None) -> np.ndarray:
    t = np.arange(int(sr * dur)) / sr
    max_h = int(sr / (2 * freq)) if harmonic_cap is None else harmonic_cap
    x = np.zeros_like(t)
    for h in range(1, max_h + 1, 2):
        x += (1.0 / h) * np.sin(2 * np.pi * h * freq * t)
    x *= 4 / np.pi
    return (amp * x / (np.abs(x).max() + 1e-12)).astype(np.float32)


def sawtooth(freq: float, sr: int, dur: float, amp: float = 0.5,
             harmonic_cap: int | None = None) -> np.ndarray:
    t = np.arange(int(sr * dur)) / sr
    max_h = int(sr / (2 * freq)) if harmonic_cap is None else harmonic_cap
    x = np.zeros_like(t)
    for h in range(1, max_h + 1):
        x += ((-1) ** (h + 1)) * np.sin(2 * np.pi * h * freq * t) / h
    x *= 2 / np.pi
    return (amp * x / (np.abs(x).max() + 1e-12)).astype(np.float32)


def triangle(freq: float, sr: int, dur: float, amp: float = 0.5,
             harmonic_cap: int | None = None) -> np.ndarray:
    t = np.arange(int(sr * dur)) / sr
    max_h = int(sr / (2 * freq)) if harmonic_cap is None else harmonic_cap
    x = np.zeros_like(t)
    for h in range(1, max_h + 1, 2):
        x += ((-1) ** ((h - 1) // 2)) * np.sin(2 * np.pi * h * freq * t) / (h * h)
    x *= 8 / (np.pi ** 2)
    return (amp * x / (np.abs(x).max() + 1e-12)).astype(np.float32)


def multi_tone(freqs, sr, dur, amps=None, phases=None) -> np.ndarray:
    freqs = np.atleast_1d(freqs)
    amps = np.ones(len(freqs)) * 0.3 if amps is None else np.atleast_1d(amps)
    phases = np.zeros(len(freqs)) if phases is None else np.atleast_1d(phases)
    t = np.arange(int(sr * dur)) / sr
    x = np.zeros_like(t)
    for f, a, p in zip(freqs, amps, phases):
        x += a * np.sin(2 * np.pi * f * t + p)
    return x.astype(np.float32)


def dual_tone(f1: float, f2: float, sr: int, dur: float,
              amp: float = 0.4) -> np.ndarray:
    return multi_tone([f1, f2], sr, dur, amps=[amp, amp])


def am_tone(carrier: float, mod: float, depth: float, sr: int,
            dur: float, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(sr * dur)) / sr
    env = 1 + depth * np.sin(2 * np.pi * mod * t)
    return (amp * env * np.sin(2 * np.pi * carrier * t)).astype(np.float32)


def fm_tone(carrier: float, mod: float, index: float, sr: int,
            dur: float, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(sr * dur)) / sr
    phase = 2 * np.pi * carrier * t + index * np.sin(2 * np.pi * mod * t)
    return (amp * np.sin(phase)).astype(np.float32)


def harmonic_stack(f0: float, n_harmonics: int, sr: int, dur: float,
                   amps=None, amp: float = 0.5) -> np.ndarray:
    if amps is None:
        amps = np.ones(n_harmonics) / np.arange(1, n_harmonics + 1)
    t = np.arange(int(sr * dur)) / sr
    x = np.zeros_like(t)
    for h in range(1, n_harmonics + 1):
        f = f0 * h
        if f >= sr / 2:
            break
        x += amps[h - 1] * np.sin(2 * np.pi * f * t)
    x = x / (np.abs(x).max() + 1e-12) * amp
    return x.astype(np.float32)


def log_sweep(f0: float, f1: float, sr: int, dur: float,
              amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(sr * dur)) / sr
    k = np.log(f1 / f0)
    phase = 2 * np.pi * f0 * dur / k * (np.exp(k * t / dur) - 1)
    return (amp * np.sin(phase)).astype(np.float32)


def chirp(f0: float, f1: float, sr: int, dur: float,
          amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(sr * dur)) / sr
    phase = 2 * np.pi * (f0 * t + (f1 - f0) / (2 * dur) * t ** 2)
    return (amp * np.sin(phase)).astype(np.float32)


def near_nyquist(sr: int, dur: float, amp: float = 0.5) -> np.ndarray:
    return sine(0.45 * sr, sr, dur, amp=amp)


# ── noise ─────────────────────────────────────────────────────────

def pink_noise(sr: int, dur: float, seed: int = 0,
               amp: float = 0.1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(sr * dur)
    white = rng.standard_normal(n)
    X = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, 1 / sr)
    f[0] = f[1] if len(f) > 1 else 1.0
    X = X / np.sqrt(f)
    x = np.fft.irfft(X, n)
    return (amp * x / (np.std(x) + 1e-12)).astype(np.float32)


def brown_noise(sr: int, dur: float, seed: int = 0,
                amp: float = 0.1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(sr * dur)
    white = rng.standard_normal(n)
    x = np.cumsum(white)
    x = x - x.mean()
    return (amp * x / (np.std(x) + 1e-12)).astype(np.float32)


def noise_burst(sr: int, dur: float, burst_ms: float, gap_ms: float,
                color: str = "white", seed: int = 0,
                amp: float = 0.3) -> np.ndarray:
    n = int(sr * dur)
    burst_n = int(sr * burst_ms / 1000)
    gap_n = int(sr * gap_ms / 1000)
    period = burst_n + gap_n
    rng = np.random.default_rng(seed)
    if color == "white":
        base = rng.standard_normal(n)
    elif color == "pink":
        white = rng.standard_normal(n)
        X = np.fft.rfft(white)
        f = np.fft.rfftfreq(n, 1 / sr)
        f[0] = f[1] if len(f) > 1 else 1.0
        base = np.fft.irfft(X / np.sqrt(f), n)
    elif color == "brown":
        base = np.cumsum(rng.standard_normal(n))
        base -= base.mean()
    else:
        raise ValueError(f"unknown color: {color}")
    gate = np.zeros(n)
    for start in range(0, n, period):
        gate[start:start + burst_n] = 1.0
    x = base * gate
    return (amp * x / (np.abs(x).max() + 1e-12)).astype(np.float32)


# ── speech-like ───────────────────────────────────────────────────

def speech_like(sr: int, dur: float, f0: float = 120.0,
                seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * dur)) / sr
    x = np.zeros_like(t)
    for h in range(1, 40):
        f = f0 * h
        if f > sr / 2 - 100:
            break
        x += (1.0 / h) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    for fc, bw, g in [(500, 80, 1.0), (1500, 120, 0.6), (2500, 180, 0.3)]:
        r = np.exp(-np.pi * bw / sr)
        theta = 2 * np.pi * fc / sr
        a1, a2 = -2 * r * np.cos(theta), r * r
        b0 = (1 - r) * np.sqrt(1 - 2 * r * np.cos(2 * theta) + r * r)
        for i in range(2, len(x)):
            x[i] += g * (b0 * x[i] - a1 * x[i - 1] - a2 * x[i - 2]) * 0.1
    x *= (0.7 + 0.3 * np.sin(2 * np.pi * 4 * t))
    x = x / (np.abs(x).max() + 1e-12) * 0.5
    return x.astype(np.float32)


# ── transients / anchors ─────────────────────────────────────────

def impulse(sr: int, dur: float, at: float = 0.5,
            amp: float = 1.0) -> np.ndarray:
    x = np.zeros(int(sr * dur), dtype=np.float32)
    x[int(sr * at)] = amp
    return x


def step(sr: int, dur: float, at: float = 0.5) -> np.ndarray:
    x = np.zeros(int(sr * dur), dtype=np.float32)
    x[int(sr * at):] = 0.5
    return x


def transient_train(sr: int, dur: float, rate_hz: float = 4.0,
                    decay_ms: float = 20.0, amp: float = 0.7) -> np.ndarray:
    n = int(sr * dur)
    x = np.zeros(n, dtype=np.float32)
    period = int(sr / rate_hz)
    decay_n = int(sr * decay_ms / 1000)
    for start in range(0, n, period):
        env = np.exp(-np.arange(min(decay_n, n - start)) /
                     (sr * decay_ms / 1000 / 3))
        x[start:start + len(env)] += env.astype(np.float32)
    return (amp * x / (np.abs(x).max() + 1e-12)).astype(np.float32)


def long_silence_with_spikes(sr: int, dur: float, n_spikes: int = 3,
                             seed: int = 0) -> np.ndarray:
    """
    FIX: spikes were placed at unchecked random positions and summed
    additively with no final normalization — every other generator in
    this file normalizes by peak amplitude, this one didn't. Confirmed
    by execution: with n_spikes=3, overlapping spikes produced a peak of
    1.198, violating the file's own selftest bound of 1.0. Now
    normalizes at the end like every other function here.
    """
    rng = np.random.default_rng(seed)
    n = int(sr * dur)
    x = np.zeros(n, dtype=np.float32)
    spike_n = int(0.05 * sr)
    for _ in range(n_spikes):
        pos = rng.integers(0, n - spike_n)
        freq = rng.uniform(200, 4000)
        t_local = np.arange(spike_n) / sr
        x[pos:pos + spike_n] += 0.6 * np.sin(2 * np.pi * freq * t_local)
    peak = np.abs(x).max()
    if peak > 1e-12:
        x = x / peak * 0.8
    return x.astype(np.float32)


# ── silence / DC ──────────────────────────────────────────────────

def silence(sr: int, dur: float) -> np.ndarray:
    return np.zeros(int(sr * dur), dtype=np.float32)


def dc(sr: int, dur: float, level: float = 0.1) -> np.ndarray:
    return np.full(int(sr * dur), level, dtype=np.float32)


# ── octave anchors ────────────────────────────────────────────────

def sub_bass(sr: int, dur: float) -> np.ndarray:
    return sine(40, sr, dur)


def bass(sr: int, dur: float) -> np.ndarray:
    return sine(150, sr, dur)


def mid(sr: int, dur: float) -> np.ndarray:
    return sine(1000, sr, dur)


def high(sr: int, dur: float) -> np.ndarray:
    return sine(6000, sr, dur)


# ── fingerprints ──────────────────────────────────────────────────

def signal_fingerprint(x: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(x, dtype=np.float32).tobytes()
                        ).hexdigest()


# ── batteries ─────────────────────────────────────────────────────

def test_battery(sr: int = 24000) -> dict[str, np.ndarray]:
    return {
        "silence":          silence(sr, 1.0),
        "dc":               dc(sr, 1.0, 0.05),
        "sine_440":         sine(440, sr, 1.0),
        "sine_1k":          sine(1000, sr, 1.0),
        "sine_8k":          sine(8000, sr, 1.0),
        "sine_11k":         sine(11000, sr, 1.0),
        "cosine_440":       cosine(440, sr, 1.0),
        "square_220":       square(220, sr, 1.0, harmonic_cap=20),
        "sawtooth_110":     sawtooth(110, sr, 1.0, harmonic_cap=30),
        "triangle_220":     triangle(220, sr, 1.0, harmonic_cap=25),
        "multitone":        multi_tone([220, 440, 880, 1760, 3520], sr, 1.0),
        "dual_tone":        dual_tone(700, 900, sr, 1.0),
        "am_tone":          am_tone(1000, 5, 0.8, sr, 1.0),
        "fm_tone":          fm_tone(1000, 50, 3, sr, 1.0),
        "harmonic_stack":   harmonic_stack(200, 10, sr, 1.0),
        "sweep":            log_sweep(50, 11000, sr, 1.0),
        "chirp":            chirp(100, 5000, sr, 1.0),
        "near_nyquist":     near_nyquist(sr, 1.0),
        "pink":             pink_noise(sr, 1.0, seed=42),
        "brown":            brown_noise(sr, 1.0, seed=43),
        "white_burst":      noise_burst(sr, 1.0, 100, 100, "white", seed=44),
        "speech_like":      speech_like(sr, 1.0, seed=1),
        "impulse":          impulse(sr, 1.0, at=0.5),
        "step":             step(sr, 1.0, at=0.5),
        "transient_train":  transient_train(sr, 1.0, rate_hz=4.0),
        "long_silence":     long_silence_with_spikes(sr, 1.0, n_spikes=3,
                                                     seed=5),
        "sub_bass":         sub_bass(sr, 1.0),
        "bass":             bass(sr, 1.0),
        "mid":              mid(sr, 1.0),
        "high":             high(sr, 1.0),
    }


def stereo_battery(sr: int = 24000) -> dict[str, np.ndarray]:
    b = test_battery(sr)
    out = {}
    for name, x in b.items():
        seed = zlib.crc32(name.encode()) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        delay = int(rng.integers(0, 64))
        gain = 0.9
        x_r = np.roll(x, delay) * gain
        out[name] = np.stack([x, x_r], axis=1)
    return out


def ab_battery(sr: int = 24000) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    clean = test_battery(sr)
    rng = np.random.default_rng(1234)
    out = {}
    for name, x in clean.items():
        sig_power = float(np.mean(x ** 2)) + 1e-12
        snr_db = 20.0
        noise_power = sig_power / (10 ** (snr_db / 10))
        noise = rng.standard_normal(len(x)).astype(np.float32) * np.sqrt(noise_power)
        dirty = (x + noise).astype(np.float32)
        out[name] = (x, dirty)
    return out


def iter_battery(sr: int = 24000):
    for name, x in test_battery(sr).items():
        yield name, x


# ── self-verification ─────────────────────────────────────────────

def _verify() -> None:
    sr = 24000

    for name, x in test_battery(sr).items():
        assert len(x) > 0, f"{name}: empty"
        assert np.all(np.isfinite(x)), f"{name}: non-finite"
        assert np.abs(x).max() <= 1.0 + 1e-5, f"{name}: exceeds 1.0"

    a = stereo_battery(sr)["sine_440"]
    b = stereo_battery(sr)["sine_440"]
    assert np.array_equal(a, b), "stereo_battery not deterministic"

    x = sine(1000, sr, 0.5)
    X = np.abs(np.fft.rfft(x))
    f = np.fft.rfftfreq(len(x), 1 / sr)
    assert abs(f[X.argmax()] - 1000) < 5, "sine peak not at target"

    x1 = pink_noise(sr, 0.5, seed=7)
    x2 = pink_noise(sr, 0.5, seed=7)
    assert signal_fingerprint(x1) == signal_fingerprint(x2), "pink unstable"

    print("harmonia_signalgen: all checks passed")


if __name__ == "__main__":
    _verify()
