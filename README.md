# desync-detect

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://github.com/habib-analyst/desync-detect)
[![GitHub stars](https://img.shields.io/github/stars/habib-analyst/desync-detect.svg)](https://github.com/habib-analyst/desync-detect)

A CPU-only audio-video desynchronization detector — a triage signal for analysts working with suspected deepfakes and cheapfakes. It correlates the audio energy envelope with the visual motion-energy envelope and flags where the two stop agreeing.

## Quickstart

```bash
pip install -r requirements.txt        # numpy, scipy, opencv-python-headless
pip install -e .                      # installs the `desync-detect` CLI

# See it work end-to-end on synthetic samples (offline, ~10 s):
python -m desyncdetect demo

# Generate test samples:
desync-detect synth --out ./samples/

# Analyze a real video (extract audio first — OpenCV can't demux audio):
ffmpeg -i video.mp4 -ar 16000 -ac 1 audio.wav
desync-detect analyze video.mp4 --audio audio.wav --window 5 --out report.json
```

## Measured results

Actual output of `python -m desyncdetect demo` (three synthetic samples, analyzed live):

```
sample     verdict      median|lag|     slope  notes
------------------------------------------------------------------------------
clean      IN_SYNC             0 ms       +0  median |lag| 0 ms within ±150 ms
lagged     DESYNCED          400 ms       +0  2/2 windows exceed ±150 ms
drift      DRIFT             460 ms      +47  lag drifts +47 ms/s over the clip

[OK ] clean: got IN_SYNC, expected IN_SYNC
[OK ] lagged: got DESYNCED, expected DESYNCED
[OK ] drift: got DRIFT, expected DRIFT
```

Tests: 19 tests, no network — envelope extraction on synthetic signals, cross-correlation recovering known planted lags (±1 frame), verdict logic on synthetic sync/lag/drift inputs, silent/static → `INCONCLUSIVE`, CLI round-trips, and the demo as a smoke test (`python -m pytest tests/ -v`).

## How it works

1. **Envelopes**: audio → per-video-frame RMS energy; video → mean absolute frame-difference (motion energy), both sampled at the video frame rate.
2. **Sync analysis**: normalized cross-correlation per sliding window gives a lag estimate in ms (positive = audio lags video) and a confidence score from peak prominence. A greedy continuity pass disambiguates alias peaks across windows; a Theil–Sen robust fit of lag-vs-time detects drift.
3. **Verdict**: `IN_SYNC` (median |lag| ≤ 150 ms), `DESYNCED` (sustained |lag| above tolerance), `DRIFT` (lag slope > 30 ms/s with > 250 ms total swing), `INCONCLUSIVE` (flat envelopes — silent or static input).

It won't convict a video — it tells an analyst *where to look*.

## Honest limitations

- **Triage signal, not a deepfake verdict.** Dubbed content, artistic edits, and variable-frame-rate phone footage also produce DESYNCED flags. It prioritizes attention; it does not prove manipulation.
- **Fails on static scenes and silent audio.** No motion or no sound means nothing to correlate; the tool reports `INCONCLUSIVE` rather than guessing.
- **Coarse.** It detects gross sync errors (≥ ~150 ms), not the 1–2 frame subtleties a phoneme-viseme model would catch.
- **Needs a separate WAV** — OpenCV can't demux audio tracks; extract with one `ffmpeg` line (above) or use `--from-npz`.

## License

MIT © 2026 Habib Ur Rehman

---
Built by [Habib Ur Rehman](https://github.com/habib-analyst) · ML/AI Engineer & healthcare-AI researcher ·
[LinkedIn](https://www.linkedin.com/in/hur-dev) · [habib.top](https://habib.top)
