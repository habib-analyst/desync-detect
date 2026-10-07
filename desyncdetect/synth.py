"""Synthetic A/V test-sample generator (fully offline, deterministic).

Generates tiny mp4 videos (cv2.VideoWriter) paired with WAV audio tracks:

- ``clean``   — motion flashes and audio beeps aligned (expect IN_SYNC)
- ``lagged``  — audio delayed by a constant ``lag_ms`` (expect DESYNCED)
- ``drift``   — audio delay grows over time (expect DRIFT)

The visual "motion" is a white square pulsing for a single frame on a black
background; the audio is 40 ms 880 Hz clicks. Both are brief, sharp pulses,
so the audio-energy and motion-energy envelopes match closely — exactly what
the envelope-correlation detector needs.
"""

from __future__ import annotations

import math
import os
import wave

import cv2
import numpy as np

SAMPLE_RATE = 16000
BEEP_FREQ = 880.0
BEEP_DUR = 0.04           # seconds: one video frame at 25 fps. Events are
                          # brief, sharp pulses (click + single-frame flash)
                          # so the audio and motion envelopes match closely.
BEEP_OFFSET = 0.5        # first beep at t=0.5 s
SIZE = 64                # 64x64 frames
SQUARE = 24              # flashing square side length (px)


def _beep_wave(n: int) -> np.ndarray:
    """One beep: sine with raised-cosine edges (no clicks)."""
    t = np.arange(n) / SAMPLE_RATE
    w = np.sin(2 * math.pi * BEEP_FREQ * t)
    edge = min(n, int(0.01 * SAMPLE_RATE))
    ramp = np.ones(n)
    ramp[:edge] = 0.5 - 0.5 * np.cos(np.pi * np.arange(edge) / edge)
    ramp[-edge:] = ramp[:edge][::-1]
    return (w * ramp).astype(np.float64)


def render_audio(path: str, beep_times: list[float], duration: float) -> None:
    n = int(duration * SAMPLE_RATE)
    sig = np.zeros(n, dtype=np.float64)
    beep = _beep_wave(int(BEEP_DUR * SAMPLE_RATE))
    for t in beep_times:
        i = int(round(t * SAMPLE_RATE))
        j = min(n, i + len(beep))
        if i < n:
            sig[i:j] += beep[: j - i]
    sig = np.clip(sig, -1.0, 1.0)
    pcm = (sig * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def render_video(path: str, flash_times: list[float], duration: float, fps: float) -> None:
    n_frames = int(round(duration * fps))
    # each event is a single-frame pulse in the frame whose interval contains
    # the event onset: flash frame = floor(t * fps). This keeps the visual
    # event centered on its timestamp (no systematic quantization bias).
    flash_frames = {int(math.floor(t * fps)) for t in flash_times}
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(path, fourcc, fps, (SIZE, SIZE))
    if not vw.isOpened():
        raise RuntimeError(f"could not open VideoWriter for {path}")
    c0 = (SIZE - SQUARE) // 2
    for i in range(n_frames):
        frame = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        if i in flash_frames:
            frame[c0:c0 + SQUARE, c0:c0 + SQUARE] = 255
        vw.write(frame)
    vw.release()


def _onsets(duration: float, seed: int = 7) -> list[float]:
    """Deterministic pseudo-random beep onsets (intervals in [0.7, 1.3] s).

    Randomized (but seeded) intervals avoid the lag-aliasing ambiguity of a
    perfectly periodic beep train: cross-correlation then has one dominant
    peak instead of a comb of equal peaks every period.
    """
    rng = np.random.default_rng(seed)
    t, out = BEEP_OFFSET, []
    while True:
        if t + BEEP_DUR > duration:
            break
        out.append(t)
        t += float(rng.uniform(0.7, 1.3))
    return out


def synthesize(out_dir: str, fps: float = 25.0, lag_ms: float = 400.0,
               drift_ms_per_s: float = 50.0) -> dict[str, dict[str, str]]:
    """Write clean/lagged/drift sample pairs; return {name: {video, audio}}."""
    os.makedirs(out_dir, exist_ok=True)
    samples: dict[str, dict[str, str]] = {}

    # clean: flashes and beeps aligned
    dur = 6.0
    onsets = _onsets(dur)
    _write_pair(out_dir, "clean", onsets, onsets, dur, fps)
    samples["clean"] = _paths(out_dir, "clean")

    # lagged: audio delayed by lag_ms (audio lags behind video)
    lag_s = lag_ms / 1000.0
    _write_pair(out_dir, "lagged", onsets, [t + lag_s for t in onsets], dur, fps)
    samples["lagged"] = _paths(out_dir, "lagged")

    # drift: audio delay grows linearly with time
    dur = 16.0
    onsets = _onsets(dur)
    rate = drift_ms_per_s / 1000.0
    _write_pair(out_dir, "drift", onsets, [t + rate * t for t in onsets], dur, fps)
    samples["drift"] = _paths(out_dir, "drift")

    return samples


def _paths(out_dir: str, name: str) -> dict[str, str]:
    return {
        "video": os.path.join(out_dir, f"{name}.mp4"),
        "audio": os.path.join(out_dir, f"{name}.wav"),
    }


def _write_pair(out_dir: str, name: str, flash_times: list[float],
                beep_times: list[float], duration: float, fps: float) -> None:
    p = _paths(out_dir, name)
    render_video(p["video"], flash_times, duration, fps)
    render_audio(p["audio"], beep_times, duration)
