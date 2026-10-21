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
