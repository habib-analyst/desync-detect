# desync-detect

[![CI](https://github.com/habib-analyst/desync-detect/actions/workflows/ci.yml/badge.svg)](https://github.com/habib-analyst/desync-detect/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/habib-analyst/desync-detect/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://github.com/habib-analyst/desync-detect)

A fast, CPU-only **audio-video desynchronization detector** — a triage signal for analysts working with suspected deepfakes and cheapfakes.

## Problem statement

Many manipulated videos — cheapfakes, lip-sync deepfakes, re-encoded rips — carry a subtle but measurable defect: the **audio track drifts out of sync with the visual motion** (clock drift between capture devices, sloppy dubbing, frame-rate conversions, or synthesis artifacts). Deep neural detectors are accurate but heavy: they need GPUs, large models, and labeled data.

`desync-detect` takes the opposite approach: a **classical signal-processing triage tool** that runs in seconds on a CPU with three light dependencies. It correlates the *audio energy envelope* with the *visual motion-energy envelope* and flags segments where the two stop agreeing. It won't convict a video — but it tells an analyst *where to look*.

## How it works

```
video.mp4 ──┬──► audio energy envelope ──┐
            │   (per-frame RMS of WAV)   │
            │                            ▼
            │                    normalized cross-correlation
            │                    over sliding windows
            │                            │
            └──► motion energy envelope ─┘   (mean |frame diff|)
                                             │
                        ┌────────────────────┴────────────────────┐
                        │ per-window lag (ms) + sync confidence   │
                        │ alias disambiguation across windows     │
                        └────────────────────┬────────────────────┘
                                             ▼
                        verdict: IN_SYNC / DESYNCED / DRIFT / INCONCLUSIVE
                        + suspect segments with timestamps
```

1. **Envelopes** (`features.py`): audio → per-video-frame RMS energy; video → mean absolute frame-difference (motion energy). Both sampled at the video frame rate.
2. **Sync analysis** (`sync.py`): normalized cross-correlation per sliding window gives a lag estimate (ms; positive = audio lags video) and a confidence score from peak prominence. Quasi-periodic content (speech rhythm) creates alias peaks one period apart, so a greedy continuity pass disambiguates lags across windows. A Theil–Sen robust fit of lag-vs-time detects **drift** even when one window aliases.
3. **Verdict**: `IN_SYNC` (median |lag| ≤ 150 ms), `DESYNCED` (sustained |lag| above tolerance), `DRIFT` (lag slope > 30 ms/s with > 250 ms total swing), `INCONCLUSIVE` (flat audio/motion envelopes — silent or static input).
4. **Synthetic samples** (`synth.py`): deterministic generation of clean / constant-lag / drifting A/V pairs, so tests and the demo run fully offline.

**Dependency choice:** `opencv-python-headless` (not `imageio-ffmpeg`) for video I/O — it ships prebuilt wheels, needs no system FFmpeg, and works headless in CI. OpenCV cannot decode audio tracks, so `analyze` takes the audio as a separate WAV (`ffmpeg -i video.mp4 -ar 16000 -ac 1 audio.wav`); a `--from-npz` mode accepts precomputed envelopes directly.

## Quickstart (< 5 min)

```bash
pip install -r requirements.txt        # numpy, scipy, opencv-python-headless
pip install -e .                      # installs the `desync-detect` CLI

# 1. see it work end-to-end on synthetic samples (offline, ~10 s)
python -m desyncdetect demo

# 2. generate test samples
desync-detect synth --out ./samples/

# 3. analyze a real video (extract audio first: OpenCV can't read audio tracks)
ffmpeg -i video.mp4 -ar 16000 -ac 1 audio.wav
desync-detect analyze video.mp4 --audio audio.wav --window 5 --out report.json
```

## Real example output

`python -m desyncdetect demo` — three synthetic samples, analyzed live (actual output):

```
desync-detect 0.1.0 — demo (synthetic samples, offline)

sample     verdict      median|lag|     slope  notes
------------------------------------------------------------------------------
clean      IN_SYNC             0 ms       +0  median |lag| 0 ms within ±150 ms
lagged     DESYNCED          400 ms       +0  2/2 windows exceed ±150 ms
drift      DRIFT             460 ms      +47  lag drifts +47 ms/s over the clip

[OK ] clean: got IN_SYNC, expected IN_SYNC
[OK ] lagged: got DESYNCED, expected DESYNCED
[OK ] drift: got DRIFT, expected DRIFT

demo passed: clean→IN_SYNC, lagged→DESYNCED, drift→DRIFT
```

`desync-detect analyze lagged.mp4 --audio lagged.wav --window 4 --step 2` (planted lag: +400 ms):

```
verdict: DESYNCED  (median |lag| 400 ms, 2/2 valid windows)
  [  0.0s -   4.0s] lag  +400.0 ms  conf 0.79
  [  2.0s -   6.0s] lag  +400.0 ms  conf 0.69
  suspect segment: 0.0s - 6.0s (median lag +400 ms)
  note: 2/2 windows exceed ±150 ms
```

## Honest limitations

- **This is a triage signal, not a deepfake verdict.** A `DESYNCED` flag means "audio and motion envelopes disagree here" — that happens in real videos too (dubbed content, artistic edits, variable-frame-rate phone footage). It prioritizes analyst attention; it does not prove manipulation.
- **Fails on static scenes and silent audio.** With no motion or no sound there is nothing to correlate: the tool reports `INCONCLUSIVE` rather than guessing. Talking-head footage with a static background is the hardest case.
- **Envelope correlation is coarse.** It detects gross sync errors (≥ ~150 ms), not the 1–2 frame subtleties a phoneme-viseme model would catch.
- **Quasi-periodic content can alias.** Rhythmic speech/motion produces correlation peaks one period apart; per-window lags are disambiguated by continuity, and the drift trend fit is robust to isolated jumps, but pathological rhythms can still fool individual windows.
- **Needs a separate WAV.** OpenCV can't demux audio; you must extract it (one `ffmpeg` line, see above) or use `--from-npz`.

## Roadmap

- [ ] Optional `imageio-ffmpeg` audio demux so `analyze video.mp4` works without manual extraction
- [ ] Per-face lip-region motion envelopes (MediaPipe) instead of full-frame motion energy
- [ ] Adaptive window sizing for short clips
- [ ] HTML report with envelope + lag plots

## Tests

```bash
python -m pytest tests/ -v
```

19 tests, no network: envelope extraction on synthetic signals, cross-correlation recovering known planted lags (±1 frame), verdict logic on synthetic sync/lag/drift inputs, silent/static → `INCONCLUSIVE`, CLI round-trips, and the demo as a smoke test. Videos are tiny (64×64, a few seconds) so CI stays fast.

## Citations & related work

- Suwajanakorn et al., "Lip Sync DeepFakes" — early demonstration that audio-driven lip synthesis leaves sync artifacts.
- Prajwal et al., **Wav2Lip** (ACM MM 2020) — the flip side: how accurate lip-sync is synthesized, and what its failure modes look like.
- Agarwal et al., "Detecting Deep-Fake Videos from Phoneme-Viseme Mismatches" (CVPRW 2020) — phoneme/viseme alignment as a manipulation signal; this tool is a coarse, model-free cousin of that idea.
- Mittal et al., "Emotions Don't Lie" (ACM MM 2020) — multimodal (audio+video) affect inconsistencies for deepfake detection.
- Korshunov & Marcel, "Vulnerability Assessment of Lip-Based Biometric Verification" — on the fragility of lip-sync under manipulation.

## License

MIT — © 2026 Habib Ur Rehman. See [LICENSE](LICENSE).
