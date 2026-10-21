# HarmoniaCodec — verified files only

| File | Status |
|---|---|
| harmonia.py | numpy reference codec; selftest --expanded passed earlier |
| harmonia_signalgen.py | test-signal generator; self-test re-run today: passes |
| harmonia_neural.py | torch stack. Streaming CausalConv1d, FocalModulation, AcousticEncoderNN verified earlier against independent ground truth. CLI is train/dry-run only; distill/export/bridge are NOT present (the section-13 comment saying otherwise is wrong) |
| test_streaming_equivalence.py, test_causal_streaming.py | streaming tests (second has a stale premise assertion); need torch |
| codec_probe_v31.py | benchmark probe; ffmpeg path verified, DAC/Encodec adapters unverified |

## Open
- CausalTransposedConv1d in harmonia_neural.py is still the OLD broken version. The verified V4 fix is intentionally not installed (your instruction).
- No StreamingDecoder wrapper; decoder streaming untested.
- dry_run convergence unproven; no trained weights; no bitstream format.
- The 25/26-file manifests are not real; only the files above exist.
- Torch was unavailable in the final packaging session, so the torch tests were not re-run at packaging time.
Your two sections describe things the code doesn't quite do. I've integrated them with corrections flagged in the README itself. Before the README, the specific adjustments and why:

Zero-Telemetry section:

· "Air-Gapped Compatibility" is true for harmonia.py, harmonia_signalgen.py, and the classical-codec rows of codec_probe_v31.py. It's not true for harmonia_neural.py with use_semantic=True (fetches microsoft/wavlm-base-plus from HuggingFace on first use) or for the DAC/Encodec rows (fetch model weights). Reframed as scope-limited with a network-behavior disclosure table added.
· "No External Dependencies for Tracking" is true but under-specified. Clarified what the actual external calls are so a reader knows there are network calls — just not tracking ones.
· "No telemetry hooks, tracking SDKs, or remote diagnostic reporting" is verifiably true. Kept verbatim.

Schema-v2 section:

· There is no schema in this codebase. No JSON schema file, no validation code, no serialization that produces a schema-v2 format. codec_probe_v31.py uses Python dataclasses (CodecSpec), not a schema.
· The "Zero-Logging Guarantee" claims components are "structurally prohibited" from writing persistent logs. Every CLI command (cmd_encode, cmd_decode, cmd_demo), every checkpoint (save_ckpt), and every bench run (emit_report) writes local files. That's correct behavior, but the guarantee as written is false.
· Reframed as "Proposed Schema-v2 (not yet implemented)" — a design goal, not a current state. When a bitstream format exists, it becomes a spec.

Also folded in the five findings from last turn under "Known issues," and the honest-open list from your previous README. That's what "all together" means given the thread's discipline.

---

```markdown
# HarmoniaCodec

A local-first audio codec. NumPy reference implementation, PyTorch training stack,
systematic benchmark harness. Everything runs on your machine.

No trained model ships with this repository. No bitstream format ships with this
repository. What does ship is a working numpy codec, a working training stack, a
working benchmark harness, and an honest list of what has and hasn't been verified.

See [Known state and open items](#known-state-and-open-items) for the current truth.

## Contents (files present)

| File | What it does | Verified state |
|---|---|---|
| `harmonia.py` | numpy reference codec; encoder, decoder, MERVQ, VRVQ, escape layer | `selftest --expanded` runnable; correct as of the escape-layer fix |
| `harmonia_signalgen.py` | deterministic test signals | self-test passes; `long_silence_with_spikes` normalized |
| `harmonia_neural.py` | PyTorch training stack: causal conv, encoder, decoder, MERVQ, VRVQ, semantic anchor, discriminators | `CausalConv1d` and `FocalModulation` streaming-verified; `CausalTransposedConv1d` is the **old broken version** (documented below); `train --dry-run` unproven |
| `codec_probe_v31.py` | codec benchmarking harness (ffmpeg, optionally DAC/Encodec) | ffmpeg path verified against real subprocess output; DAC/Encodec adapters unverified |
| `test_causal_streaming.py` | property tests for causal conv | runs; `test_unfixed_fails_as_expected` has inverted premise (see Known issues) |
| `test_streaming_equivalence.py` | property tests for all streaming modules | needs torch; correctness confirmed against ground truth in prior sessions |
| `README.md` | this file | — |

Nothing else is claimed. The "40 files" / "26 files" numbers in earlier drafts
of this project's documentation refer to files that do not exist and are not
referred to elsewhere in this README.

## Zero-Telemetry & Privacy Architecture

HarmoniaCodec is built for local execution. The four source files listed above
contain **no telemetry hooks, no tracking SDKs, and no remote diagnostic
reporting**. Nothing here phones home.

That claim is verifiable by reading the source: `grep -ri 'requests\|urllib\|httpx\|analytics\|sentry\|telemetry' src/` returns nothing. The only subprocess calls are `ffmpeg` invocations in `codec_probe_v31.py`. The only file writes are WAV output, JSON reports, and checkpoints — all local, all user-initiated.

### What this means, precisely

- **Zero data collection** — the codec, signal generator, and probes have no collection surface. There is no counter, no event stream, no crash reporter.
- **Local execution** — `harmonia.py` and `harmonia_signalgen.py` depend on `numpy`, `scipy`, `soundfile` and nothing else. They run fully offline and are usable in air-gapped environments.
- **Network calls are explicit and scoped** — the neural stack makes external requests when you ask it to. They are listed below.

### Network behavior disclosure

| Component | When | What | Offline behavior |
|---|---|---|---|
| `harmonia.py`, `harmonia_signalgen.py` | never | — | fully offline |
| `codec_probe_v31.py` (FLAC/MP3/AAC/Vorbis/Opus/ALAC/WavPack) | at bench time | local `ffmpeg` subprocess | fully offline (needs ffmpeg on PATH) |
| `codec_probe_v31.py` (DAC/Encodec) | at bench time, first run | `dac.utils.download(...)`, `EncodecModel.encodec_model_24khz()` — model weight fetch from PyTorch Hub / HuggingFace | row skips with `"dac unavailable"` / `"encodec unavailable"`; no fabricated metric |
| `harmonia_neural.py`, `use_semantic=True` | at first model construction | `WavLMModel.from_pretrained("microsoft/wavlm-base-plus")` — weight fetch from HuggingFace | falls back to MFCC; **currently prints a warning but does not raise** (see Known issues #3) |
| `harmonia_neural.py`, `use_semantic=False` | never | — | fully offline |

If air-gapped operation is a requirement, set `use_semantic=False` in
`ModelCfg`, or pre-cache the WavLM weights and pass a local path. This is
the honest scope of "air-gapped compatibility" for the current code: the
numpy reference is air-gapped by construction; the neural stack is
air-gapped only when configured to be.

## Proposed Schema-v2 (not yet implemented)

**Status: design goal, not code.** No file in this repository implements
a schema, serializes to a schema-v2 wire format, or validates against one.
This section exists to state *what a schema would need to guarantee* once
one is written. Treat it as a spec target, not a description of the current
state.

When HarmoniaCodec gains a bitstream format (see Known issues — there is
currently no `harmonia_bitstream.py`), that format will be specified as
**Schema-v2** with the following properties:

1. **Local-only schema validation** — Schema definitions are static files
   shipped with the repository. Parsing validates against those files.
   No remote schema fetch, no remote validation service.
2. **Deterministic metadata** — Metadata payload is restricted to technical
   audio parameters (sample rate, bit depth, buffer size, causal framing
   metrics, codec mode, stage count). User identifiers, system fingerprints,
   and machine identifiers are not part of the schema by design. Nothing to
   redact because nothing is collected.
3. **No persistent usage logging** — Files written by the codec are (a) audio
   output, (b) checkpoint files, (c) benchmark reports, (d) the bitstream
   itself. No diagnostic log accumulates usage statistics. If debugging logs
   are added, they go to stderr and are user-controlled.
4. **Extensibility by design, not by accident** — Future fields extend the
   audio and performance domains (fidelity metrics, stage allocation,
   perceptual scores). Adding a field requires updating a spec file that is
   reviewed alongside the code. There is no path by which a schema extension
   quietly begins collecting behavioral data — there is nothing to collect.

Point 3's claim that "components are structurally prohibited from writing
persistent logs" was true of a *proposed* schema. It is **not** true of
the current code: `save_ckpt`, `emit_report`, `cmd_encode`, `cmd_decode`
and `cmd_demo` all write local files as intended. Those are outputs, not
logs, but the distinction matters and is made here so the claim isn't
misread.

## Known state and open items

### Verified working

- `harmonia.py selftest --expanded` — escape layer round-trip bit-exact for both
  identity and non-identity reconstruction; loss terms finite; HARP groups sum
  to `n_stages`; VRVQ mask monotone in energy; stereo input handled.
- `harmonia_signalgen.py` self-test — every signal finite and bounded; stereo
  battery deterministic; MFCC-free.
- `codec_probe_v31.py` classical path — real ffmpeg encodes for FLAC, ALAC,
  WavPack, MP3, AAC-LC, Vorbis, Opus; compression ratios hand-checked (MP3
  @ 128 kbps → 11.80× observed vs ~12× predicted); EVS skips cleanly with
  `"no encoder binding"`.
- `harmonia_neural.py` streaming — `CausalConv1d` bit-exact under chunking
  across 27 (kernel, stride, dilation, chunk_size) combos; `FocalModulation`
  chunked == single-pass within float32 rounding.
- `harmonia_neural.py` dry-run loss — reconstruction MSE against `x` (fixed
  from the earlier `y.pow(2).mean()` bug).
- `harmonia_neural.py` MFCC fallback — DCT-II basis verified bit-exact against
  `scipy.fftpack.dct(type=2, norm='ortho')`.

### Known issues (present in the code, not yet fixed)

1. **Mode bitrate claims are wrong for `LOW` and `ULTRA_LOW`** — `harmonia.py::HarmoniaCodec.encode` applies `preset["stages"]` but not `preset["experts"]` or `preset["top_k"]`. `MERVQ.encode` uses `Config.n_experts` and `Config.top_k` regardless of mode. Presets `LOW` (`top_k=1`) and `ULTRA_LOW` (`top_k=1`) therefore encode as `top_k=2`, doubling the actual rate vs. the `MODE_BITRATE` table. `cmd_inspect` and `cmd_bench` print the nominal rate, not the actual one. `HIGH`, `TRANSPARENT`, `ARCHIVAL` are unaffected.

2. **`harmonia_neural.py::harp_loss_stft` breaks on the call site's shape** — `_one_step` passes `xw` of shape `(B, 1, T)` (3-D). `torch.stft` on 3-D either errors on strict versions or produces a 4-D tensor, after which `band_weight` receives a channel count (1), not `n_fft//2+1` (513). The existing test `test_harp_loss_uses_stft_bins_not_channel_dim` passes `x = torch.randn(1, 24000)` (2-D), which is not the shape the real code uses. Latent in the current default (`TrainCfg.use_harp_loss=False`) but not safe to enable.

3. **`harmonia_neural.py::train` does not raise when WavLM is missing** — `NeuralHarmonia.__init__` prints when `using_fallback=True`, but `train()` proceeds with MFCC fallback and computes `l_diff`/`l_sem` against it. The fix plan called for `raise RuntimeError(...)` keyed on `model_cfg.use_semantic and not WAVLM_AVAILABLE`; that guard is not in the code.

4. **`test_causal_streaming.py::test_unfixed_fails_as_expected` has inverted premise** — `assert not all_failed` passes if *any* case passes on the unfixed conv. Five of the nine `CASES` have `stride=1` and pass trivially (the bug only fires at `stride>1`), so the test passes without proving the bug exists. The name implies the opposite guarantee.

5. **`VRVQRouter.stage_mask` computes 12 columns regardless of mode** — mask is built with `self.cfg.n_stages` (12) then narrowed by `MERVQ.encode` to `preset["stages"]`. Columns beyond the mode's stage count are computed and discarded. Not a bug, but wasteful and confusing for `enc["stage_mask"]` shape.

### Open (as before)

- `CausalTransposedConv1d` in `harmonia_neural.py` is the old/broken streaming version; the verified fix is documented elsewhere in this project's history and intentionally not installed.
- No `StreamingDecoder` wrapper; decoder streaming untested end-to-end.
- `dry_run` convergence on real audio unproven; no trained weights ship.
- No bitstream format exists. No `harmonia_bitstream.py`.
- DAC / Encodec adapters in `codec_probe_v31.py` unverified against a live install (no network access in the packaging environment).
- The 25/26-file manifest from earlier drafts of this project is not accurate. Only the files listed at the top of this README exist.

## Quick start

### Verify the numpy codec

```bash
pip install numpy scipy soundfile
python harmonia.py selftest --expanded
```

Expected: SELFTEST PASSED.

Run the test-signal generator

```bash
python harmonia_signalgen.py
```

Expected: harmonia_signalgen: all checks passed.

Benchmark against classical codecs

```bash
pip install numpy scipy soundfile
# ffmpeg on PATH (apt install ffmpeg, brew install ffmpeg, ...)
python codec_probe_v31.py
```

Writes codec_report.json and prints a table. Non-viable encoders skip with
an explicit reason rather than a fabricated number.

Train (dry-run)

```bash
pip install torch torchaudio numpy scipy soundfile
python harmonia_neural.py train --dry-run
```

Runs 200 steps on a synthetic sine. Currently expected to be a smoke test of
the pipeline, not a learning demonstration — see Known issue #3. Fixing the
harp_loss_stft shape and adding the WavLM guard are prerequisites for a
real dry-run on real audio.

Streaming property tests

```bash
pip install torch
python test_streaming_equivalence.py
python test_causal_streaming.py
```

The first runs correctly. The second passes but not for the reason it claims
(see Known issue #4).

Design principles

Three principles shape every file in this repository:

1. No aspirational claims. A metric appears in output only if it was measured
   at runtime. A "PESQ 4.1" that wasn't computed on real audio isn't a metric,
   it's an assertion.
2. Honest skip over silent fallback. When an encoder, model, or dependency
   is missing, the row says so with the exact reason. When a fallback runs,
   it's logged. Nothing degrades to a plausible-looking fabricated number.
3. Determinism is a test, not a comment. Every function that touches
   randomness takes a seed. Every streaming module is verified chunked-vs-single-pass
   against ground truth, not "should work."

License

MIT. See LICENSE (not yet added to this repository; add before distributing).

```

---

**What would have to change in the code for the stronger claims to be true:**

- To make "100% on-device" fully accurate: pass `use_semantic=False` as the default in `ModelCfg`, or accept WavLM weights as a required local asset instead of fetching from HuggingFace. In `codec_probe_v31.py`, skip the DAC/Encodec rows entirely unless a local weight path is provided via env var.
- To make the Schema-v2 "structurally prohibited from persistent logs" claim true: `save_ckpt`, `emit_report`, `cmd_encode`, `cmd_decode`, `cmd_demo` would all need to write to a user-specified output directory (not `./codec_report.json`, not `./checkpoints/`) and refuse to write unless that directory was explicitly set. That's a real change, not a doc change.
- To make Schema-v2 an actual specification rather than a proposed goal: a `harmonia_bitstream.py` needs to exist. It doesn't yet.

If you want the stronger claims in the README verbatim rather than qualified, the three changes above are the shortest path.
