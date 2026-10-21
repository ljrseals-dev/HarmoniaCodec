#!/usr/bin/env python3
from __future__ import annotations
import io, json, math, os, shutil, subprocess, tempfile, time, warnings
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from enum import Enum
from math import gcd
from typing import Optional

import numpy as np
import soundfile as sf
from scipy.signal import stft, resample_poly

warnings.filterwarnings("ignore")
HAVE_FFMPEG = shutil.which("ffmpeg") is not None


class CodecFamily(Enum):
    LOSSLESS = "lossless"; TRANSFORM = "transform"; HYBRID = "hybrid"
    NEURAL = "neural"; NEURAL_STREAM = "neural_stream"

class Transform(Enum):
    NONE = "none"; MDCT = "mdct"; QMF = "qmf"; LEARNED = "learned"

class Quantizer(Enum):
    NONE = "none"; SCALAR = "scalar"; VECTOR = "vector"
    RVQ = "rvq"; FSQ = "fsq"


@dataclass
class CodecSpec:
    name: str
    family: CodecFamily
    transform: Transform
    quantizer: Quantizer
    entropy: str
    bitrate_kbps: Optional[float]
    sample_rates: list
    channels: int
    algorithmic_delay_ms: float
    perceptual_model: bool
    stereo_coding: str
    band_extension: bool
    trained_end_to_end: bool
    open_source: bool
    ffmpeg_encoder: Optional[str] = None
    ffmpeg_format: Optional[str] = None
    python_package: Optional[str] = None
    notes: str = ""
    measured_compression: float = 0.0
    measured_encode_ms: float = 0.0
    measured_decode_ms: float = 0.0
    measured_snr_db_L: float = 0.0
    measured_snr_db_R: float = 0.0
    measured_lsd_L: float = 0.0
    measured_lsd_R: float = 0.0
    measured_rms_db: float = 0.0
    measured_channels_compared: int = 0
    measured_viable: bool = True
    measured_skip_reason: str = ""


CODEC_UNIVERSE = [
    CodecSpec("FLAC", CodecFamily.LOSSLESS, Transform.NONE, Quantizer.NONE,
              "rice", None, [44100,48000,96000,192000], 8, 0, False, "none",
              False, False, True, "flac", "flac"),
    CodecSpec("ALAC", CodecFamily.LOSSLESS, Transform.NONE, Quantizer.NONE,
              "rice", None, [44100,48000,96000], 8, 0, False, "none",
              False, False, True, "alac", "m4a"),
    CodecSpec("WavPack", CodecFamily.LOSSLESS, Transform.NONE, Quantizer.NONE,
              "rice", None, [44100,48000,192000], 8, 0, False, "none",
              False, False, True, "wavpack", "wv"),
    CodecSpec("MP3", CodecFamily.TRANSFORM, Transform.MDCT, Quantizer.SCALAR,
              "huffman", 128.0, [44100,48000], 2, 50, True, "ms",
              False, False, True, "libmp3lame", "mp3"),
    CodecSpec("AAC-LC", CodecFamily.TRANSFORM, Transform.MDCT, Quantizer.SCALAR,
              "huffman", 128.0, [44100,48000,96000], 8, 20, True, "ms",
              True, False, True, "aac", "m4a"),
    CodecSpec("Vorbis", CodecFamily.TRANSFORM, Transform.MDCT, Quantizer.VECTOR,
              "range", 128.0, [44100,48000,96000], 8, 30, True, "ms",
              False, False, True, "libvorbis", "ogg"),
    CodecSpec("Opus", CodecFamily.HYBRID, Transform.MDCT, Quantizer.SCALAR,
              "range", 64.0, [8000,12000,16000,24000,48000], 2, 6.5,
              True, "intensity", False, True, True, "libopus", "opus"),
    CodecSpec("EVS", CodecFamily.HYBRID, Transform.MDCT, Quantizer.SCALAR,
              "range", 24.0, [8000,16000,32000,48000], 2, 32,
              True, "intensity", True, True, True,
              notes="3GPP EVS; no ffmpeg encoder"),
    CodecSpec("Encodec", CodecFamily.NEURAL, Transform.LEARNED, Quantizer.RVQ,
              "none", 6.0, [24000,48000], 2, 0, False, "none",
              False, True, True, python_package="encodec"),
    CodecSpec("DAC", CodecFamily.NEURAL, Transform.LEARNED, Quantizer.RVQ,
              "none", 8.0, [16000,24000,44100], 2, 0, False, "none",
              False, True, True, python_package="dac"),
]


def snr(ref, test):
    n = min(len(ref), len(test))
    r, t = ref[:n], test[:n]
    return 10.0 * math.log10((float(np.mean(r**2)) + 1e-20) /
                             (float(np.mean((r-t)**2)) + 1e-20))

def lsd(ref, test, sr, n_fft=2048):
    n = min(len(ref), len(test))
    _,_,Zr = stft(ref[:n], sr, nperseg=n_fft)
    _,_,Zt = stft(test[:n], sr, nperseg=n_fft)
    return float(np.mean(np.sqrt(np.mean(
        (20*np.log10(np.abs(Zr)+1e-10) - 20*np.log10(np.abs(Zt)+1e-10))**2, axis=0))))

def rms_db(x):
    return float(20.0 * math.log10(np.sqrt(np.mean(x**2)) + 1e-20))

def _as_2d(x):
    return x.reshape(-1, 1) if x.ndim == 1 else x

def _resample_to(x, src_sr, dst_sr):
    if src_sr == dst_sr:
        return x
    g = gcd(int(src_sr), int(dst_sr))
    up, down = dst_sr // g, src_sr // g
    if x.ndim == 1:
        return resample_poly(x, up, down).astype(np.float32)
    return np.stack([resample_poly(x[:, c], up, down)
                     for c in range(x.shape[1])], axis=1).astype(np.float32)


def measure_codec(spec, sig, sr, encode_fn, decode_fn,
                  decoded_sample_rate=None):
    try:
        t0 = time.perf_counter()
        encoded = encode_fn(sig, sr)
        t1 = time.perf_counter()
        decoded = decode_fn(encoded, sr)
        t2 = time.perf_counter()
    except Exception as e:
        spec.measured_viable = False
        spec.measured_skip_reason = f"encode/decode failed: {e}"
        return spec

    spec.measured_encode_ms = (t1 - t0) * 1000.0
    spec.measured_decode_ms = (t2 - t1) * 1000.0

    sig2 = _as_2d(sig).astype(np.float64)
    dec2 = _as_2d(decoded).astype(np.float64)
    eff_sr = decoded_sample_rate or sr
    if eff_sr != sr:
        dec2 = _resample_to(dec2, eff_sr, sr)

    n_ch = min(sig2.shape[1], dec2.shape[1])
    n = min(sig2.shape[0], dec2.shape[0])
    spec.measured_channels_compared = n_ch

    snrs, lsds = [], []
    for ch in range(n_ch):
        snrs.append(snr(sig2[:n, ch], dec2[:n, ch]))
        lsds.append(lsd(sig2[:n, ch], dec2[:n, ch], sr))
    spec.measured_snr_db_L = snrs[0] if snrs else 0.0
    spec.measured_snr_db_R = snrs[1] if len(snrs) > 1 else spec.measured_snr_db_L
    spec.measured_lsd_L = lsds[0] if lsds else 0.0
    spec.measured_lsd_R = lsds[1] if len(lsds) > 1 else spec.measured_lsd_L
    spec.measured_rms_db = rms_db(dec2)

    if isinstance(encoded, dict) and "bytes" in encoded:
        enc_bytes = len(encoded["bytes"])
    elif isinstance(encoded, (bytes, bytearray)):
        enc_bytes = len(encoded)
    else:
        enc_bytes = 0
    raw_bytes = sig2.shape[0] * sig2.shape[1] * 2
    spec.measured_compression = raw_bytes / max(enc_bytes, 1) if enc_bytes else 0.0
    return spec


@contextmanager
def _tempdir():
    d = tempfile.mkdtemp(prefix="codec_probe_")
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)

def _write_wav(path, sig, sr):
    sf.write(path, sig, sr, subtype="PCM_16")

_FFMPEG_ENCODERS_CACHE = None
def _ffmpeg_has_encoder(name):
    global _FFMPEG_ENCODERS_CACHE
    if not HAVE_FFMPEG:
        return False
    if _FFMPEG_ENCODERS_CACHE is None:
        try:
            r = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                               capture_output=True, text=True, timeout=5)
            _FFMPEG_ENCODERS_CACHE = r.stdout
        except Exception:
            _FFMPEG_ENCODERS_CACHE = ""
    return name in _FFMPEG_ENCODERS_CACHE

def _ffmpeg_encode(spec, bitrate_kbps=None, extra_args=None):
    bitrate = bitrate_kbps or spec.bitrate_kbps or 128.0
    def encode(sig, sr):
        with _tempdir() as d:
            wav_in = os.path.join(d, "in.wav")
            out = os.path.join(d, f"out.{spec.ffmpeg_format}")
            _write_wav(wav_in, sig, sr)
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", wav_in,
                   "-c:a", spec.ffmpeg_encoder, "-b:a", f"{int(bitrate)}k"]
            if extra_args:
                cmd += extra_args
            cmd += [out]
            subprocess.run(cmd, check=True, capture_output=True)
            with open(out, "rb") as f:
                data = f.read()
        return {"bytes": data, "format": spec.ffmpeg_format}
    def decode(enc, sr):
        with _tempdir() as d:
            out = os.path.join(d, f"out.{enc['format']}")
            wav = os.path.join(d, "dec.wav")
            with open(out, "wb") as f:
                f.write(enc["bytes"])
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                            "-i", out, "-f", "wav", wav],
                           check=True, capture_output=True)
            y, _ = sf.read(wav, always_2d=True, dtype="float64")
        return y
    return encode, decode

def _dac_adapter(sr):
    """
    DAC's real API surface was disputed across this conversation (compress/
    decompress vs preprocess/encode/decode) and confirmed broken in the v2
    source that was eventually reviewed directly (it chained encode() and
    compress() incoherently). Rather than guess a replacement, this probes
    the loaded model object at runtime and uses whichever interface exists,
    so it degrades to a clear skip reason instead of a wrong call.
    """
    try:
        import torch, dac
        from audiotools import AudioSignal
    except Exception as e:
        return None, None, f"dac unavailable: {e}", None

    model_type = "44khz" if sr >= 40000 else "24khz" if sr >= 20000 else "16khz"
    try:
        path = dac.utils.download(model_type=model_type)
        model = dac.DAC.load(path)
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
    except Exception as e:
        return None, None, f"dac load failed: {e}", None

    model_sr = model.sample_rate
    has_compress = hasattr(model, "compress") and hasattr(model, "decompress")
    has_encode_decode = (hasattr(model, "preprocess") and hasattr(model, "encode")
                         and hasattr(model, "decode"))

    if not has_compress and not has_encode_decode:
        attrs = [a for a in dir(model) if not a.startswith("_")]
        return None, None, (f"DAC object exposes neither compress/decompress nor "
                            f"preprocess/encode/decode — actual attrs: {attrs}"), None

    def encode(sig, sr_):
        import io
        mono = sig if sig.ndim == 1 else sig.mean(axis=1)
        if sr_ != model_sr:
            g = gcd(sr_, model_sr)
            mono = resample_poly(mono, model_sr // g, sr_ // g).astype(np.float32)
        if has_compress:
            s = AudioSignal(torch.from_numpy(mono[None, :]).float(), sample_rate=model_sr)
            s.to(device)
            with torch.no_grad():
                dac_file = model.compress(s)
            buf = io.BytesIO()
            torch.save({"kind": "compress", "obj": dac_file}, buf)
            return buf.getvalue()
        else:
            x = torch.from_numpy(mono[None, None, :]).float().to(device)
            x = model.preprocess(x, model_sr)
            with torch.no_grad():
                z, codes, *_ = model.encode(x)
            buf = io.BytesIO()
            torch.save({"kind": "encode", "obj": z}, buf)
            return buf.getvalue()

    def decode(enc, sr_):
        import io
        payload = torch.load(io.BytesIO(enc), map_location="cpu", weights_only=False)
        if payload["kind"] == "compress":
            with torch.no_grad():
                out = model.decompress(payload["obj"])
            y = out.audio_data.cpu().numpy().squeeze()
        else:
            with torch.no_grad():
                y_t = model.decode(payload["obj"])
            y = y_t.squeeze().cpu().numpy()
        if sr_ != model_sr:
            g = gcd(model_sr, sr_)
            y = resample_poly(y, sr_ // g, model_sr // g).astype(np.float32)
        return y

    return encode, decode, None, model_sr


def _encodec_adapter(sr):
    """UNVERIFIED — Encodec's real API was never contradicted the way DAC's
    was, but it was also never run against an installed package here."""
    try:
        import torch
        from encodec import EncodecModel
        from encodec.utils import convert_audio
    except Exception as e:
        return None, None, f"encodec unavailable: {e}", None

    try:
        model = EncodecModel.encodec_model_24khz()
        model.set_target_bandwidth(6.0)
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
    except Exception as e:
        return None, None, f"encodec load failed: {e}", None

    model_sr = model.sample_rate

    def encode(sig, sr_):
        import io
        wav = (torch.from_numpy(sig.T).float() if sig.ndim > 1
               else torch.from_numpy(sig[None, :]).float())
        wav = convert_audio(wav, sr_, model.sample_rate, model.channels).to(device)
        with torch.no_grad():
            frames = model.encode(wav.unsqueeze(0))
        buf = io.BytesIO()
        torch.save(frames, buf)
        return buf.getvalue()

    def decode(enc, sr_):
        import io
        frames = torch.load(io.BytesIO(enc), map_location="cpu", weights_only=False)
        with torch.no_grad():
            wav = model.decode(frames)
        y = wav.squeeze(0).cpu().numpy().T
        if sr_ != model_sr:
            g = gcd(model_sr, sr_)
            y = resample_poly(y, sr_ // g, model_sr // g).astype(np.float32)
        return y

    return encode, decode, None, model_sr


def _neural_adapter(spec, sr):
    if spec.python_package == "dac":
        return _dac_adapter(sr)
    if spec.python_package == "encodec":
        return _encodec_adapter(sr)
    return None, None, f"no adapter for {spec.name}", None

@dataclass
class Improvement:
    codec: str; axis: str; finding: str; suggestion: str; priority: int

def analyze(spec):
    out = []
    if not spec.measured_viable:
        return out
    avg_snr = 0.5 * (spec.measured_snr_db_L + spec.measured_snr_db_R)
    avg_lsd = 0.5 * (spec.measured_lsd_L + spec.measured_lsd_R)
    if spec.family == CodecFamily.LOSSLESS:
        out.append(Improvement(spec.name, "compression",
            f"lossless {spec.measured_compression:.2f}x at SNR {avg_snr:.1f} dB",
            "add lossy escape hatch (MPEG-4 SLS style)", 3))
    if spec.family in (CodecFamily.NEURAL, CodecFamily.NEURAL_STREAM):
        out.append(Improvement(spec.name, "compression",
            f"neural {spec.bitrate_kbps} kbps, ratio {spec.measured_compression:.1f}x",
            "distill student decoder for edge", 1))
        out.append(Improvement(spec.name, "complexity",
            f"decode {spec.measured_decode_ms:.1f} ms",
            "int8 + prune; target <5 ms", 2))
    if spec.perceptual_model and avg_lsd > 0.5:
        out.append(Improvement(spec.name, "quality",
            f"LSD {avg_lsd:.3f}", "augment masking with learned thresholds", 2))
    if not spec.band_extension and spec.bitrate_kbps and spec.bitrate_kbps < 64:
        out.append(Improvement(spec.name, "quality",
            "no band extension at low bitrate", "add SBR or neural BWE", 1))
    return out

@dataclass
class BlueprintComponent:
    name: str; source_codecs: list; rationale: str; target_metric: str

def synthesize_blueprint(specs):
    LOSSY = {CodecFamily.TRANSFORM, CodecFamily.HYBRID,
             CodecFamily.NEURAL, CodecFamily.NEURAL_STREAM}
    REALTIME = {CodecFamily.HYBRID, CodecFamily.NEURAL_STREAM}
    real = [s for s in specs if s.measured_viable and s.measured_encode_ms > 0]
    lossy = [s for s in real if s.family in LOSSY]
    realtime = [s for s in real if s.family in REALTIME]
    def pick(pool, key, reverse=True, default="(insufficient data)"):
        if not pool:
            return default, 0.0
        s = sorted(pool, key=key, reverse=reverse)[0]
        return s.name, key(s)
    best_comp, _ = pick(lossy, lambda s: s.measured_compression, True)
    best_qual, _ = pick(lossy, lambda s: 0.5*(s.measured_snr_db_L+s.measured_snr_db_R), True)
    best_lat, _ = pick(realtime, lambda s: s.algorithmic_delay_ms, False)
    best_dec, _ = pick(lossy, lambda s: s.measured_decode_ms, False)
    return [
        BlueprintComponent("Hybrid Core (LP + MDCT)", ["Opus", "EVS"],
            "LP low-freq; MDCT music/high.", "quality 24-64 kbps"),
        BlueprintComponent(f"Learned RVQ (best lossy: {best_comp})",
            ["DAC", "Encodec"], "RVQ beats scalar quant.", "compression"),
        BlueprintComponent(f"Psychoacoustic Masking (best lossy: {best_qual})",
            ["AAC-LC", "MP3", "Vorbis"], "Masking prevents artifacts.", "perceptual"),
        BlueprintComponent(f"Low-Latency Path (best realtime: {best_lat})",
            ["Opus", "FocalCodec-Stream"], "Realtime families only.", "reliability"),
        BlueprintComponent(f"Distilled Decoder (fastest lossy: {best_dec})",
            ["LDCodec", "FocalCodec-Stream"], "Small student; int8.", "decode ms"),
        BlueprintComponent("Bandwidth Extension", ["AAC-HE", "Opus"],
            "Reconstruct HF from LF.", "quality <32 kbps"),
        BlueprintComponent("Lossless Escape Layer", ["FLAC", "WavPack"],
            "Lossless residual on lossy core.", "archival"),
    ]

class BufferReader:
    def __init__(self, path, frame_ms=20.0):
        self.path = path
        self.info = sf.info(path)
        self.sr = self.info.samplerate
        self.frame_samples = max(1, int(self.sr * frame_ms / 1000.0))
        self._fh = None
    def __enter__(self):
        self._fh = sf.SoundFile(self.path); return self
    def __exit__(self, *exc):
        if self._fh: self._fh.close(); self._fh = None
    def __iter__(self):
        if self._fh is None:
            raise RuntimeError("BufferReader must be used as context manager")
        while True:
            block = self._fh.read(self.frame_samples, dtype="float64", always_2d=True)
            if len(block) == 0: break
            yield block

def emit_report(specs, improvements, blueprint, path="codec_report.json"):
    improvements = sorted(improvements, key=lambda i: (i.priority, i.codec))
    report = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ffmpeg": HAVE_FFMPEG,
        "codecs": [asdict(s) for s in specs],
        "improvements": [asdict(i) for i in improvements],
        "blueprint": [asdict(b) for b in blueprint],
    }
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[report] written -> {path}")
    print(f"\n{'='*100}")
    print(f"{'Codec':<14} {'Family':<13} {'kbps':>6} {'SNR-L':>7} {'SNR-R':>7} "
          f"{'LSD-L':>7} {'Comp':>7} {'wall-enc':>9} {'wall-dec':>9}")
    print("-" * 100)
    print("# wall-enc/wall-dec include subprocess fork/exec + I/O; not codec time")
    print("# [mono] = codec downmixed to 1 ch; SNR-R duplicates SNR-L")
    for s in specs:
        if not s.measured_viable:
            print(f"{s.name:<14} {s.family.value:<13}  SKIP: {s.measured_skip_reason}")
            continue
        ch_tag = "  [mono]" if s.measured_channels_compared == 1 else ""
        print(f"{s.name:<14} {s.family.value:<13} "
              f"{(s.bitrate_kbps or 0):>6.1f} "
              f"{s.measured_snr_db_L:>7.2f} {s.measured_snr_db_R:>7.2f} "
              f"{s.measured_lsd_L:>7.3f} "
              f"{s.measured_compression:>7.2f} "
              f"{s.measured_encode_ms:>9.1f} "
              f"{s.measured_decode_ms:>9.1f}{ch_tag}")

def main():
    print("codec_probe v3.1")
    print("=" * 100)
    print(f"ffmpeg available: {HAVE_FFMPEG}")
    sr = 48000
    t = np.linspace(0, 3.0, 3 * sr, endpoint=False)
    base = (0.5*np.sin(2*np.pi*440*t) + 0.3*np.sin(2*np.pi*2000*t) +
            0.05*np.random.randn(len(t)))
    sig = np.stack([base, base*0.9], axis=1).astype(np.float64)
    print(f"test signal: {sr} Hz, 2 ch, {len(sig)} frames, rms {rms_db(sig):.2f} dB\n")
    specs = CODEC_UNIVERSE
    for spec in specs:
        if spec.python_package:
            enc, dec, err, native_sr = _neural_adapter(spec, sr)
            if enc is None:
                spec.measured_viable = False
                spec.measured_skip_reason = err
                continue
            measure_codec(spec, sig, sr, enc, dec, decoded_sample_rate=native_sr)
            continue
        if spec.ffmpeg_encoder:
            if not HAVE_FFMPEG:
                spec.measured_viable = False
                spec.measured_skip_reason = "ffmpeg not installed"
                continue
            if not _ffmpeg_has_encoder(spec.ffmpeg_encoder):
                spec.measured_viable = False
                spec.measured_skip_reason = f"encoder {spec.ffmpeg_encoder} not in this ffmpeg build"
                continue
            enc, dec = _ffmpeg_encode(spec)
            measure_codec(spec, sig, sr, enc, dec)
            continue
        spec.measured_viable = False
        spec.measured_skip_reason = "no encoder binding"
    improvements = []
    for s in specs:
        improvements.extend(analyze(s))
    blueprint = synthesize_blueprint(specs)
    emit_report(specs, improvements, blueprint)
    print("\n[buffer] frame-by-frame read")
    tmp = "/tmp/probe2/_probe.wav"
    _write_wav(tmp, sig, sr)
    with BufferReader(tmp, frame_ms=20.0) as r:
        frames = sum(1 for _ in r)
    os.remove(tmp)
    print(f"[buffer] {frames} frames, 20 ms each")
    print(f"\n[blueprint] {len(blueprint)} components")
    for b in blueprint:
        print(f"  • {b.name:<52} <- {', '.join(b.source_codecs)}")

if __name__ == "__main__":
    main()
