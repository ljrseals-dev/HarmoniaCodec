#!/usr/bin/env python3
"""
harmonia.py — HarmoniaCodec, single-file runnable reference.

No optimizer, no training — inference only. Train in harmonia_neural.py,
bridge weights in with export_numpy_bridge.

Usage:
    python harmonia.py selftest [--expanded]
    python harmonia.py demo
    python harmonia.py info
    python harmonia.py inspect --mode good
    python harmonia.py bench
    python harmonia.py encode in.wav out.harm --mode good [--stream]
    python harmonia.py decode out.harm out.wav [--length N]
"""
from __future__ import annotations
import argparse, math, os, pickle, struct, sys, time
from dataclasses import dataclass
import numpy as np


class Mode:
    ARCHIVAL = "archival"; TRANSPARENT = "transparent"; HIGH = "high"
    GOOD = "good"; LOW = "low"; ULTRA_LOW = "ultra_low"; EXTREME = "extreme"


MODE_PRESETS = {
    Mode.ARCHIVAL:    dict(stages=12, experts=4, top_k=2, gen=False, lossless=True),
    Mode.TRANSPARENT: dict(stages=12, experts=4, top_k=2, gen=False, lossless=False),
    Mode.HIGH:        dict(stages=9,  experts=4, top_k=2, gen=False, lossless=False),
    Mode.GOOD:        dict(stages=6,  experts=4, top_k=1, gen=False, lossless=False),
    Mode.LOW:         dict(stages=3,  experts=2, top_k=1, gen=True,  lossless=False),
    Mode.ULTRA_LOW:   dict(stages=1,  experts=2, top_k=1, gen=True,  lossless=False),
    Mode.EXTREME:     dict(stages=0,  experts=1, top_k=1, gen=True,  lossless=False),
}

MODE_BITRATE = {
    Mode.ARCHIVAL: 1024.0, Mode.TRANSPARENT: 128.0, Mode.HIGH: 32.0,
    Mode.GOOD: 8.0, Mode.LOW: 2.0, Mode.ULTRA_LOW: 0.65, Mode.EXTREME: 0.20,
}


@dataclass
class Config:
    sr: int = 24000
    enc_dim: int = 128
    latent_dim: int = 64
    codebook_size: int = 1024
    codebook_dim: int = 64
    n_stages: int = 12  # must be >= max(preset['stages'] for preset in MODE_PRESETS.values());
                        # MERVQ and VRVQRouter allocate exactly this many stage
                        # slots at construction, and per-mode encode() only ever
                        # narrows to a smaller subset — it cannot widen beyond
                        # what was built here. Confirmed by execution: the
                        # previous default of 9 crashed on 'archival' and
                        # 'transparent' modes (both request 12 stages).
    n_experts: int = 4
    top_k: int = 2
    semantic_dim: int = 768
    semantic_codebook_size: int = 512
    harp_groups: int = 3
    diffusion_steps: int = 8
    w_recon: float = 1.0
    w_psycho: float = 0.5
    w_sem: float = 0.3
    w_commit: float = 0.25
    w_balance: float = 0.1
    w_otcfm: float = 0.2


class PsychoacousticMasking:
    def __init__(self, sr: int, n_fft: int = 1024):
        self.sr, self.n_fft = sr, n_fft
        self.freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        self.bark = self._hz_to_bark(self.freqs)
        self.ath = self._ath(self.freqs)
        b = self.bark[:, None] - self.bark[None, :]
        S = 10 ** (2.5 - 0.5 * np.abs(b))
        self.spread = S / (S.sum(1, keepdims=True) + 1e-12)

    @staticmethod
    def _hz_to_bark(f):
        f = np.maximum(f, 1e-6)
        return 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)

    @staticmethod
    def _ath(f):
        fk = np.maximum(f, 20.0) / 1000.0
        return (3.64 * fk ** -0.8 - 6.5 * np.exp(-0.6 * (fk - 3.3) ** 2)
                + 1e-3 * fk ** 4)

    def _frame(self, x, n_fft: int | None = None):
        n = n_fft or self.n_fft
        w = np.hanning(n)
        if len(x) >= n:
            return x[:n] * w
        return np.pad(x, (0, n - len(x))) * w

    def threshold(self, x, n_fft: int | None = None) -> np.ndarray:
        n = n_fft or self.n_fft
        if n != self.n_fft:
            m = PsychoacousticMasking(self.sr, n)
            return m.threshold(x)
        X = np.fft.rfft(self._frame(x))
        p = (np.abs(X) ** 2) / (self.n_fft ** 2) + 1e-20
        p_db = 10 * np.log10(p)
        mask_db = p_db @ self.spread + self.ath - 10.0
        return 10 ** (mask_db / 10.0)


class HARPPartitioner:
    def __init__(self, n_stages: int, n_groups: int = 3):
        self.n_stages, self.n_groups = n_stages, n_groups
        base, rem = divmod(n_stages, n_groups)
        sizes = [base + (1 if i < rem else 0) for i in range(n_groups)]
        self.group_sizes = sizes
        self.stage_to_group = []
        for g, sz in enumerate(sizes):
            self.stage_to_group.extend([g] * sz)

    def band_weights(self, stage: int, n_bins: int) -> np.ndarray:
        g = self.stage_to_group[stage]
        pos = sum(1 for s in range(stage) if self.stage_to_group[s] == g)
        sz = max(1, self.group_sizes[g])
        center = (g + (pos + 0.5) / sz) / self.n_groups
        idx = np.linspace(0, 1, n_bins)
        sigma = 1.0 / (self.n_groups * 1.5)
        w = np.exp(-0.5 * ((idx - center) / sigma) ** 2)
        return np.maximum(w, 1e-9)


class Codebook:
    def __init__(self, size: int, dim: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.size, self.dim = size, dim
        self.embed = rng.standard_normal((size, dim)).astype(np.float32) * 0.1
        self.usage_step = np.zeros(size, dtype=np.float32)
        self.usage_cumulative = np.zeros(size, dtype=np.float32)

    def nearest(self, x):
        d = ((x[:, None, :] - self.embed[None, :, :]) ** 2).sum(-1)
        idx = d.argmin(-1)
        np.add.at(self.usage_step, idx, 1.0)
        np.add.at(self.usage_cumulative, idx, 1.0)
        return self.embed[idx], idx

    def balance_loss(self) -> float:
        if self.usage_step.sum() == 0:
            return 0.0
        p = self.usage_step / self.usage_step.sum()
        return float(np.sum(p * np.log((p + 1e-12) / (1.0 / self.size))))

    def utilization(self) -> float:
        return float((self.usage_cumulative > 0).mean())

    def step_reset(self):
        self.usage_step[:] = 0.0
        self.usage_cumulative *= 0.999


class R3VQRefiner:
    def __init__(self, dim: int, scale: float = 0.5):
        self.dim, self.scale = dim, scale

    def refine(self, x_est, x_orig):
        return x_est + self.scale * (x_orig - x_est)


class MERVQ:
    def __init__(self, cfg: Config, seed: int = 0):
        self.cfg = cfg
        self.stages = [
            [Codebook(cfg.codebook_size, cfg.codebook_dim, seed + 1000 * s + e)
             for e in range(cfg.n_experts)]
            for s in range(cfg.n_stages)
        ]
        self.refiners = [R3VQRefiner(cfg.codebook_dim) for _ in range(cfg.n_stages)]
        self.harp = HARPPartitioner(cfg.n_stages, cfg.harp_groups)

    def _route(self, x):
        if x.size == 0:
            return [0]
        v = x.mean(0)
        n = np.linalg.norm(v) + 1e-12
        v = v / n
        scores = [float(np.dot(v, np.roll(v, e + 1)))
                  for e in range(self.cfg.n_experts)]
        return np.argsort(scores)[::-1][: self.cfg.top_k].tolist()

    def encode(self, z, stage_mask=None):
        T, D = z.shape
        residual = z.copy()
        refined = np.zeros_like(z)
        tokens = []
        for s in range(self.cfg.n_stages):
            if stage_mask is not None and not stage_mask[:, s].any():
                tokens.append({"expert": [0] * T, "index": [0] * T, "skip": True})
                continue
            experts = self._route(residual)
            cands = []
            for e in experts:
                q, idx = self.stages[s][e].nearest(residual)
                cands.append((q, idx, e))
            stack_q = np.stack([c[0] for c in cands], 1)
            dists = ((stack_q - residual[:, None, :]) ** 2).sum(-1)
            best = dists.argmin(-1)
            q = stack_q[np.arange(T), best]
            if stage_mask is not None:
                m = stage_mask[:, s][:, None].astype(np.float32)
                q = q * m
            chosen_e = [cands[int(b)][2] for b in best]
            chosen_idx = [int(cands[int(b)][1][t]) for t, b in enumerate(best)]
            tokens.append({"expert": chosen_e, "index": chosen_idx, "skip": False})
            refined = self.refiners[s].refine(refined + q, z)
            residual = z - refined
        return {"tokens": tokens, "T": T, "D": D}

    def decode(self, enc, stage_mask=None):
        T, D = enc["T"], enc["D"]
        out = np.zeros((T, D), dtype=np.float32)
        for s, tok in enumerate(enc["tokens"]):
            if tok.get("skip"):
                continue
            idx = np.asarray(tok["index"])
            exp = np.asarray(tok["expert"])
            for e in range(self.cfg.n_experts):
                m = exp == e
                if m.any():
                    out[m] += self.stages[s][e].embed[idx[m]]
            if stage_mask is not None:
                out = out * stage_mask[:, s][:, None].astype(np.float32)
        return out

    def balance_loss(self):
        return float(np.mean([cb.balance_loss()
                              for st in self.stages for cb in st]))

    def utilization(self):
        return float(np.mean([cb.utilization()
                              for st in self.stages for cb in st]))

    def step_reset(self):
        for st in self.stages:
            for cb in st:
                cb.step_reset()


class VRVQRouter:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.min_stages = 1

    def importance(self, z):
        e = (z ** 2).sum(-1)
        return (e / (e.max() + 1e-12)).astype(np.float32)

    def stage_mask(self, z):
        imp = self.importance(z)
        T = imp.shape[0]; n = self.cfg.n_stages
        if n == 0:
            return np.zeros((T, 0), dtype=bool)
        k = np.clip(np.round(self.min_stages + imp * (n - self.min_stages)),
                    self.min_stages, n).astype(int)
        mask = np.zeros((T, n), dtype=bool)
        for t in range(T):
            mask[t, : k[t]] = True
        return mask


class OTCFMRegularizer:
    def __init__(self, cfg): self.cfg = cfg
    def loss(self, z1):
        z0 = np.random.randn(*z1.shape).astype(np.float32)
        t = np.random.rand(z1.shape[0], 1).astype(np.float32)
        zt = (1 - t) * z0 + t * z1
        v_target = z1 - z0
        return float(np.mean((v_target - v_target.mean(0)) ** 2))


class SemanticPath:
    def __init__(self, cfg: Config, seed: int = 0):
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        self.proj = rng.standard_normal(
            (cfg.latent_dim, cfg.semantic_dim)).astype(np.float32)
        self.proj /= np.linalg.norm(self.proj, axis=0, keepdims=True) + 1e-12
        self.codebook = rng.standard_normal(
            (cfg.semantic_codebook_size, cfg.semantic_dim)).astype(np.float32)
        self.codebook /= np.linalg.norm(
            self.codebook, axis=-1, keepdims=True) + 1e-12

    def features(self, z):
        return np.tanh(z @ self.proj)

    def quantize(self, feats):
        fn = feats / (np.linalg.norm(feats, axis=-1, keepdims=True) + 1e-12)
        idx = (fn @ self.codebook.T).argmax(-1)
        return self.codebook[idx], idx


class AcousticEncoder:
    def __init__(self, cfg: Config, seed: int = 0):
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        self.n_fft, self.hop = 1024, 256
        self.win = np.hanning(self.n_fft).astype(np.float32)
        self.down = rng.standard_normal(
            (self.n_fft // 2 + 1, cfg.latent_dim)).astype(np.float32)
        self.down /= np.sqrt(self.n_fft // 2 + 1)

    def _frames(self, x):
        if x.ndim == 2: x = x.mean(1)
        n = 1 + max(0, (len(x) - self.n_fft) // self.hop)
        out = np.zeros((n, self.n_fft), dtype=np.float32)
        for i in range(n):
            out[i] = x[i * self.hop: i * self.hop + self.n_fft] * self.win
        return out

    def encode(self, x):
        frames = self._frames(x)
        spec = np.fft.rfft(frames, axis=1)
        mag = np.log1p(np.abs(spec)).astype(np.float32)
        return mag @ self.down


class SubBandDecoder:
    def __init__(self, cfg: Config, seed: int = 0):
        self.cfg = cfg
        self.n_fft, self.hop = 1024, 256
        self.win = np.hanning(self.n_fft).astype(np.float32)
        rng = np.random.default_rng(seed + 7)
        self.up = rng.standard_normal(
            (cfg.latent_dim, self.n_fft // 2 + 1)).astype(np.float32)
        self.up /= np.sqrt(cfg.latent_dim)

    def decode(self, z, n_samples):
        mag = np.expm1(np.maximum(z @ self.up, 0.0))
        out = np.zeros(n_samples + self.n_fft, dtype=np.float32)
        wsum = np.zeros_like(out)
        for i in range(mag.shape[0]):
            spec = mag[i].astype(np.complex64)
            frame = np.fft.irfft(spec, n=self.n_fft).astype(np.float32) * self.win
            s = i * self.hop
            out[s: s + self.n_fft] += frame
            wsum[s: s + self.n_fft] += self.win ** 2
        return (out / (wsum + 1e-8))[:n_samples]


class GenerativeRefinementHead:
    def __init__(self, cfg: Config, seed: int = 0):
        self.cfg = cfg
        rng = np.random.default_rng(seed + 11)
        self.cond_proj = rng.standard_normal(
            (cfg.semantic_dim, cfg.latent_dim)).astype(np.float32)
        self.cond_proj /= np.sqrt(cfg.semantic_dim)

    def refine(self, z, sem_tokens):
        cond = sem_tokens @ self.cond_proj
        x = z.copy()
        for s in np.linspace(1.0, 0.05, self.cfg.diffusion_steps):
            x = 0.7 * x + 0.3 * cond + np.random.randn(*x.shape
                                                       ).astype(np.float32) * s * 0.1
        return x


class LosslessEscapeLayer:
    def __init__(self, order: int = 8):
        self.order = order

    def _lpc(self, x):
        n = self.order
        r = np.correlate(x, x, "full")[len(x) - 1: len(x) + n]
        if r[0] <= 0:
            return np.zeros(n)
        a = np.zeros(n); e = r[0]
        for i in range(n):
            k = (r[i + 1] - np.dot(a[:i], r[1:i + 1][::-1])) / (e + 1e-12)
            an = a.copy(); an[i] = k; an[:i] -= k * a[:i][::-1]
            a = an; e *= (1 - k * k)
        return a

    @staticmethod
    def _rice_enc(vals, k):
        bits = []
        for v in vals.astype(np.int64):
            u = (int(v) << 1) ^ (int(v) >> 63)
            q, r = u >> k, u & ((1 << k) - 1)
            bits.extend([1] * q + [0])
            bits.extend([(r >> i) & 1 for i in range(k - 1, -1, -1)])
        out = bytearray()
        for i in range(0, len(bits), 8):
            chunk = bits[i:i + 8]; b = 0
            for x in chunk: b = (b << 1) | x
            b <<= (8 - len(chunk)); out.append(b)
        return bytes(out)

    @staticmethod
    def _rice_dec(data, n, k):
        bits = []
        for byte in data:
            for i in range(7, -1, -1): bits.append((byte >> i) & 1)
        out, i = [], 0
        while len(out) < n and i < len(bits):
            q = 0
            while i < len(bits) and bits[i] == 1: q += 1; i += 1
            i += 1
            r = 0
            for _ in range(k):
                r = (r << 1) | (bits[i] if i < len(bits) else 0); i += 1
            u = (q << k) | r
            out.append((u >> 1) ^ -(u & 1))
        return np.array(out, dtype=np.int64)

    def encode(self, orig, recon):
        n = len(recon)
        orig_i = np.clip((orig[:n] * 32767).round(), -32768, 32767).astype(np.int32)
        recon_i = np.clip((recon * 32767).round(), -32768, 32767).astype(np.int32)
        res = orig_i - recon_i

        a = self._lpc(res.astype(np.float64))
        pred = np.zeros_like(res)
        for i in range(self.order, len(res)):
            pred[i] = int(np.dot(a, res[i - self.order: i][::-1]))
        res2 = res - pred

        k = max(0, int(np.log2(np.abs(res2).mean() + 1)))
        body = self._rice_enc(res2.astype(np.int64), k)
        header = struct.pack("<II", len(res2), k) + \
                 struct.pack("<I", self.order) + \
                 a.astype(np.float64).tobytes()
        return header + body

    def decode(self, blob, recon):
        n, k = struct.unpack("<II", blob[:8])
        off = 8
        order, = struct.unpack("<I", blob[off:off + 4]); off += 4
        a = np.frombuffer(blob[off:off + 8 * order],
                          dtype=np.float64).copy(); off += 8 * order
        res2 = self._rice_dec(blob[off:], n, k)
        pred = np.zeros(n, dtype=np.int64)
        res_i = np.zeros(n, dtype=np.int64)
        for i in range(n):
            if i >= order:
                pred[i] = int(np.dot(a, res_i[i - order: i][::-1]))
            res_i[i] = pred[i] + res2[i]
        recon_i = np.clip((recon[:n] * 32767).round(),
                          -32768, 32767).astype(np.int32)
        orig_i = recon_i + res_i.astype(np.int32)
        return (orig_i.astype(np.float32) / 32767.0)


class LossBundle:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.mask = PsychoacousticMasking(cfg.sr)

    def psychoacoustic(self, x, y):
        n = min(len(x), len(y), self.mask.n_fft)
        w = np.hanning(n)
        X = np.fft.rfft(x[:n] * w); Y = np.fft.rfft(y[:n] * w)
        err = np.abs(X - Y) ** 2 + 1e-20
        thr = self.mask.threshold(x) + 1e-20
        p = np.abs(X) ** 2 + 1e-20
        prio = np.maximum(0.0, np.log10(p / thr))
        weighted = (prio * err).mean() / (prio.mean() + 1e-12)
        nmr = np.maximum(0.0, err / thr).mean()
        return float(weighted + 0.1 * nmr)

    def harp(self, powers):
        p = np.asarray(powers, dtype=np.float32)
        p = p / (p.sum() + 1e-12)
        tgt = np.linspace(1.0, 0.0, len(p)); tgt /= tgt.sum()
        return float(np.mean((p - tgt) ** 2))

    def semantic(self, a, b):
        a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
        b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
        return float(1.0 - (a * b).sum(-1).mean())


class HarmoniaCodec:
    def __init__(self, cfg: Config | None = None, seed: int = 0):
        self.cfg = cfg or Config()
        self.encoder = AcousticEncoder(self.cfg, seed)
        self.decoder = SubBandDecoder(self.cfg, seed)
        self.semantic = SemanticPath(self.cfg, seed)
        self.quantizer = MERVQ(self.cfg, seed)
        self.vrvq = VRVQRouter(self.cfg)
        self.gen = GenerativeRefinementHead(self.cfg, seed)
        self.escape = LosslessEscapeLayer()
        self.losses = LossBundle(self.cfg)
        self.otcfm = OTCFMRegularizer(self.cfg)

    def encode(self, x, mode=Mode.GOOD):
        preset = MODE_PRESETS[mode]
        t0 = time.perf_counter()
        z = self.encoder.encode(x)
        sem_feats = self.semantic.features(z)
        _, sem_idx = self.semantic.quantize(sem_feats)
        stage_mask = self.vrvq.stage_mask(z)

        saved = self.quantizer.cfg.n_stages
        self.quantizer.cfg.n_stages = preset["stages"]
        quant = self.quantizer.encode(z, stage_mask=stage_mask)
        self.quantizer.cfg.n_stages = saved

        quant["stage_mask"] = stage_mask.tolist()
        quant["semantic_idx"] = sem_idx.tolist()
        quant["mode"] = mode
        quant["_n"] = len(x)
        quant["_sr"] = self.cfg.sr

        if preset["lossless"]:
            z_hat = self.quantizer.decode(quant, stage_mask=stage_mask)
            recon = self.decoder.decode(z_hat, len(x))
            quant["lossless_blob"] = self.escape.encode(x, recon)

        quant["encode_ms"] = (time.perf_counter() - t0) * 1000.0
        return quant

    def decode(self, enc, n_samples=None):
        t0 = time.perf_counter()
        mode = enc.get("mode", Mode.GOOD)
        preset = MODE_PRESETS[mode]
        n_samples = n_samples or enc.get("_n", 0)
        stage_mask = (np.asarray(enc["stage_mask"])
                      if "stage_mask" in enc else None)

        z_q = self.quantizer.decode(enc, stage_mask=stage_mask)
        if preset["gen"]:
            sem = self.semantic.codebook[np.asarray(enc["semantic_idx"])]
            z_q = self.gen.refine(z_q, sem)
        y = self.decoder.decode(z_q, n_samples)
        if preset["lossless"] and "lossless_blob" in enc:
            y = self.escape.decode(enc["lossless_blob"], y)
        enc["decode_ms"] = (time.perf_counter() - t0) * 1000.0
        return y

    def diagnose(self, x):
        z = self.encoder.encode(x)
        sem_f = self.semantic.features(z)
        sem_t, _ = self.semantic.quantize(sem_f)
        enc = self.encode(x, Mode.GOOD)
        y = self.decode(enc, len(x))
        n = min(len(x), len(y))
        pw = []
        r = z.copy()
        for s in range(min(3, self.cfg.n_stages)):
            experts = self.quantizer._route(r)
            q, _ = self.quantizer.stages[s][experts[0]].nearest(r)
            pw.append(float((r ** 2).mean())); r = r - q
        losses = {
            "recon": float(np.mean((x[:n] - y[:n]) ** 2)),
            "psycho": self.losses.psychoacoustic(x, y),
            "sem": self.losses.semantic(sem_f, sem_t),
            "harp": self.losses.harp(pw),
            "balance": self.quantizer.balance_loss(),
            "otcfm": self.otcfm.loss(z),
        }
        losses["total"] = (
            self.cfg.w_recon * losses["recon"] +
            self.cfg.w_psycho * losses["psycho"] +
            self.cfg.w_sem * losses["sem"] +
            self.cfg.w_balance * losses["balance"] +
            self.cfg.w_otcfm * losses["otcfm"]
        )
        losses["shapes"] = {
            "x": x.shape,
            "z": z.shape,
            "sem_f": sem_f.shape,
            "enc_T": enc["T"], "enc_D": enc["D"],
            "n_tokens": len(enc["tokens"]),
            "y": y.shape,
        }
        self.quantizer.step_reset()
        return losses

    def diagnostics(self):
        return {
            "sr": self.cfg.sr,
            "stages": self.cfg.n_stages,
            "experts": self.cfg.n_experts,
            "top_k": self.cfg.top_k,
            "harp_groups": self.quantizer.harp.group_sizes,
            "harp_stage_to_group": self.quantizer.harp.stage_to_group,
            "utilization": self.quantizer.utilization(),
        }


def _snr(ref, test):
    n = min(len(ref), len(test))
    r, t = ref[:n], test[:n]
    return float(10 * math.log10((np.mean(r ** 2) + 1e-20) /
                                 (np.mean((r - t) ** 2) + 1e-20)))


def _read_audio(path, sr):
    import soundfile as sf
    x, rsr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    if rsr != sr:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(rsr, sr)
        x = resample_poly(x, sr // g, rsr // g).astype(np.float32)
    return x


def verify_roundtrip(codec: HarmoniaCodec, x: np.ndarray, mode: str) -> dict:
    enc = codec.encode(x, mode)
    y = codec.decode(enc, len(x))
    n = min(len(x), len(y))
    return {
        "mode": mode,
        "snr_db": _snr(x[:n], y[:n]),
        "enc_ms": enc["encode_ms"],
        "dec_ms": enc["decode_ms"],
        "n_tokens": len(enc["tokens"]),
        "bytes_lossless": len(enc.get("lossless_blob", b"")),
    }


def cmd_info(args):
    codec = HarmoniaCodec(Config(sr=args.sr))
    print("HarmoniaCodec")
    for k, v in codec.diagnostics().items():
        print(f"  {k}: {v}")
    print("\nModes:")
    for m, p in MODE_PRESETS.items():
        print(f"  {m:<12} {p}")


def cmd_inspect(args):
    codec = HarmoniaCodec(Config(sr=args.sr))
    preset = MODE_PRESETS[args.mode]
    harp = codec.quantizer.harp
    n_stages_used = preset["stages"]
    print(f"mode={args.mode}  stages={n_stages_used}  "
          f"experts={preset['experts']}  top_k={preset['top_k']}  "
          f"gen={preset['gen']}  lossless={preset['lossless']}")
    print(f"{'stage':>6} {'group':>6} {'band_peak':>10} {'approximate bits/frame':>24}")
    print("-" * 56)
    cb_bits = int(math.log2(codec.cfg.codebook_size))
    for s in range(n_stages_used):
        g = harp.stage_to_group[s]
        w = harp.band_weights(s, 512)
        peak_bin = int(np.argmax(w))
        bits = preset["top_k"] * cb_bits
        print(f"{s:>6} {g:>6} {peak_bin:>10} {bits:>24}")
    print(f"\nHARP groups: {harp.group_sizes}")
    print(f"stage -> group: {harp.stage_to_group[:n_stages_used]}")


def cmd_bench(args):
    codec = HarmoniaCodec(Config(sr=args.sr))
    sr = args.sr
    t = np.linspace(0, args.sec, int(sr * args.sec), endpoint=False)
    x = (0.5 * np.sin(2 * np.pi * 440 * t)
         + 0.3 * np.sin(2 * np.pi * 2000 * t)
         + 0.02 * np.random.default_rng(0).standard_normal(len(t))
         ).astype(np.float32)
    print(f"{'mode':<14} {'kbps~':>7} {'stages':>7} {'tokens':>7} "
          f"{'snr_dB':>8} {'enc_ms':>8} {'dec_ms':>8}")
    print("-" * 68)
    for mode in [Mode.TRANSPARENT, Mode.HIGH, Mode.GOOD, Mode.LOW,
                 Mode.ULTRA_LOW, Mode.EXTREME, Mode.ARCHIVAL]:
        r = verify_roundtrip(codec, x, mode)
        print(f"{mode:<14} {MODE_BITRATE[mode]:>7.2f} "
              f"{MODE_PRESETS[mode]['stages']:>7} "
              f"{r['n_tokens']:>7} {r['snr_db']:>8.2f} "
              f"{r['enc_ms']:>8.1f} {r['dec_ms']:>8.1f}")


def cmd_encode(args):
    codec = HarmoniaCodec(Config(sr=args.sr))
    x = _read_audio(args.input, args.sr)

    if args.stream:
        chunk = int(args.sr * 0.5)
        parts = []
        for i in range(0, len(x), chunk):
            seg = x[i:i + chunk]
            if len(seg) < codec.encoder.n_fft:
                seg = np.pad(seg, (0, codec.encoder.n_fft - len(seg)))
            parts.append(codec.encode(seg, args.mode))
        enc = parts[0]
        enc["_n_chunks"] = len(parts)
    else:
        enc = codec.encode(x, args.mode)

    with open(args.output, "wb") as f:
        pickle.dump(enc, f)
    print(f"encoded {len(x)} samples @ {args.sr} Hz -> {args.output}")
    print(f"  mode={args.mode} enc={enc['encode_ms']:.1f} ms"
          + (f"  chunks={enc.get('_n_chunks', 1)}" if args.stream else ""))


def cmd_decode(args):
    codec = HarmoniaCodec(Config(sr=args.sr))
    with open(args.input, "rb") as f:
        enc = pickle.load(f)
    n = args.length if args.length is not None else enc.get("_n", None)
    y = codec.decode(enc, n)
    import soundfile as sf
    sf.write(args.output, y, enc.get("_sr", args.sr))
    print(f"decoded -> {args.output}  dec={enc.get('decode_ms', 0):.1f} ms")


def cmd_demo(args):
    print("HarmoniaCodec — demo")
    print("=" * 72)
    cfg = Config(sr=24000, n_stages=6, n_experts=4, top_k=2)
    codec = HarmoniaCodec(cfg)
    print(f"config: {codec.diagnostics()}\n")
    sr = cfg.sr
    t = np.linspace(0, 3.0, 3 * sr, endpoint=False).astype(np.float32)
    x = (0.5 * np.sin(2 * np.pi * 440 * t)
         + 0.3 * np.sin(2 * np.pi * 2000 * t)
         + 0.02 * np.random.default_rng(0).standard_normal(len(t))
         ).astype(np.float32)
    print(f"{'mode':<14} {'enc ms':>8} {'dec ms':>8} {'SNR dB':>8}")
    print("-" * 72)
    for mode in [Mode.TRANSPARENT, Mode.HIGH, Mode.GOOD, Mode.LOW,
                 Mode.ULTRA_LOW, Mode.EXTREME]:
        r = verify_roundtrip(codec, x, mode)
        print(f"{mode:<14} {r['enc_ms']:>8.1f} {r['dec_ms']:>8.1f} "
              f"{r['snr_db']:>8.2f}")
    print()
    r = verify_roundtrip(codec, x, Mode.ARCHIVAL)
    print(f"archival roundtrip: SNR {r['snr_db']:.2f} dB  "
          f"blob {r['bytes_lossless'] / 1024:.1f} KB")
    print()
    losses = codec.diagnose(x)
    print("diagnostic losses (one pass, random init):")
    for k, v in losses.items():
        if k == "shapes":
            print(f"  {'shapes':<10} {v}")
        else:
            print(f"  {k:<10} {v:>12.6f}")


def cmd_selftest(args):
    print(f"Running self-test{' (expanded)' if args.expanded else ''}...")
    codec = HarmoniaCodec(Config(sr=24000))
    x = np.random.default_rng(0).standard_normal(12000).astype(np.float32) * 0.1

    for mode in MODE_PRESETS:
        enc = codec.encode(x, mode)
        y = codec.decode(enc, len(x))
        assert len(y) > 0, f"empty output for {mode}"
        assert np.all(np.isfinite(y)), f"non-finite output for {mode}"
    print("  [ok] all modes round-trip")

    blob = codec.escape.encode(x, x)
    y_rec = codec.escape.decode(blob, x)
    assert np.allclose(y_rec[: len(x)], x, atol=1e-4), \
        "escape layer not exact on identity residual"
    print("  [ok] lossless escape exact on identity residual")

    recon = x * 0.5 + 0.01 * np.random.default_rng(1
                                                    ).standard_normal(len(x)
                                                                      ).astype(np.float32)
    blob = codec.escape.encode(x, recon)
    y_rec = codec.escape.decode(blob, recon)
    assert np.allclose(y_rec[: len(x)], x, atol=1e-4), \
        "escape layer not exact when reconstruction differs"
    print("  [ok] lossless escape exact on non-identity reconstruction")

    d = codec.diagnostics()
    assert d["harp_groups"] and sum(d["harp_groups"]) == codec.cfg.n_stages
    assert len(d["harp_stage_to_group"]) == codec.cfg.n_stages
    print("  [ok] HARP groups sum to n_stages")

    thr = codec.losses.mask.threshold(x)
    assert np.all(thr > 0) and np.all(np.isfinite(thr))
    print("  [ok] masking threshold positive & finite")

    L = codec.diagnose(x)
    for k, v in L.items():
        if k == "shapes":
            assert isinstance(v, dict)
            continue
        assert np.isfinite(v), f"non-finite loss: {k}={v}"
    print("  [ok] all loss terms finite; shapes returned")

    if args.expanded:
        for s in range(codec.cfg.n_stages):
            w = codec.quantizer.harp.band_weights(s, 512)
            assert w.max() > 0, f"stage {s} has all-zero band weights"
        print("  [ok] expanded: no all-zero band weight stages")

        z_lo = np.ones((8, codec.cfg.latent_dim), dtype=np.float32) * 0.01
        z_hi = np.ones((8, codec.cfg.latent_dim), dtype=np.float32) * 1.0
        m_lo = codec.vrvq.stage_mask(z_lo).sum()
        m_hi = codec.vrvq.stage_mask(z_hi).sum()
        assert m_hi >= m_lo, "VRVQ mask not monotone in energy"
        print("  [ok] expanded: VRVQ mask monotone in energy")

        xs = np.stack([x, x * 0.9], axis=1)
        enc = codec.encode(xs, Mode.GOOD)
        ys = codec.decode(enc, len(xs))
        assert np.all(np.isfinite(ys)), "stereo input broke codec"
        print("  [ok] expanded: stereo input handled")

    print("SELFTEST PASSED")


def main():
    p = argparse.ArgumentParser(prog="harmonia")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("info"); pi.add_argument("--sr", type=int, default=24000)
    pi.set_defaults(fn=cmd_info)

    pins = sub.add_parser("inspect")
    pins.add_argument("--mode", default=Mode.GOOD, choices=list(MODE_PRESETS))
    pins.add_argument("--sr", type=int, default=24000)
    pins.set_defaults(fn=cmd_inspect)

    pbench = sub.add_parser("bench")
    pbench.add_argument("--sr", type=int, default=24000)
    pbench.add_argument("--sec", type=float, default=3.0)
    pbench.set_defaults(fn=cmd_bench)

    pe = sub.add_parser("encode")
    pe.add_argument("input"); pe.add_argument("output")
    pe.add_argument("--mode", default=Mode.GOOD, choices=list(MODE_PRESETS))
    pe.add_argument("--sr", type=int, default=24000)
    pe.add_argument("--stream", action="store_true")
    pe.set_defaults(fn=cmd_encode)

    pd = sub.add_parser("decode")
    pd.add_argument("input"); pd.add_argument("output")
    pd.add_argument("--sr", type=int, default=24000)
    pd.add_argument("--length", type=int, default=None)
    pd.set_defaults(fn=cmd_decode)

    pdemo = sub.add_parser("demo"); pdemo.set_defaults(fn=cmd_demo)
    pst = sub.add_parser("selftest")
    pst.add_argument("--expanded", action="store_true")
    pst.set_defaults(fn=cmd_selftest)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
