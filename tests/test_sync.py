"""Sync-analysis tests: known planted lags recovered within ±1 frame."""

import numpy as np
import pytest

from desyncdetect.sync import analyze_sync, cross_correlation_lag, Verdict

FPS = 25.0
FRAME_MS = 1000.0 / FPS  # 40 ms


def burst_train(n, period=25, width=3, shift=0, jitter=0.0, seed=0):
    """Deterministic pseudo-random burst train (stand-in for envelopes)."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    pos = 10 + shift
    while pos + width < n:
        h = 1.0 + jitter * (rng.random() - 0.5)
        x[pos:pos + width] = h
        pos += period
    return x


def test_cross_correlation_recovers_planted_lag():
    motion = burst_train(250, seed=1)
    for shift_frames in (0, 5, 10):
        audio = np.roll(motion, shift_frames)  # audio delayed by shift_frames
        lag_ms, conf = cross_correlation_lag(audio, motion, FPS)
        assert abs(lag_ms - shift_frames * FRAME_MS) <= FRAME_MS + 1e-9, \
            f"shift={shift_frames}: got {lag_ms} ms"
        assert conf > 0.8


def test_cross_correlation_sign_convention():
    # positive lag_ms == audio LAGS BEHIND video
    motion = burst_train(250, seed=2)
    audio = np.roll(motion, 8)
    lag_ms, _ = cross_correlation_lag(audio, motion, FPS)
    assert lag_ms > 0
    audio_lead = np.roll(motion, -8)
    lag_ms2, _ = cross_correlation_lag(audio_lead, motion, FPS)
    assert lag_ms2 < 0


def test_cross_correlation_flat_signal_zero_confidence():
    lag_ms, conf = cross_correlation_lag(np.zeros(100), burst_train(100), FPS)
    assert conf == 0.0


def test_verdict_in_sync():
    motion = burst_train(300, seed=3)
    audio = motion.copy()
    rep = analyze_sync(audio, motion, FPS, window_sec=4.0, step_sec=2.0)
    assert rep.verdict == Verdict.IN_SYNC
    assert rep.n_valid >= 2


def test_verdict_desynced_constant_lag():
    motion = burst_train(300, seed=4)
    audio = np.roll(motion, 10)  # 400 ms constant lag
    rep = analyze_sync(audio, motion, FPS, window_sec=4.0, step_sec=2.0)
    assert rep.verdict == Verdict.DESYNCED
    assert 300 < rep.median_lag_ms < 500
    assert len(rep.suspect_segments) >= 1


def test_verdict_drift():
    # lag grows 0 -> 600 ms across the clip
    n = 400
    motion = burst_train(n, seed=5)
    audio = np.zeros(n)
    for i in range(n):
        shift = int(round(i / n * 15))  # up to 15 frames = 600 ms
        src = i - shift
        audio[i] = motion[src] if src >= 0 else 0.0
    rep = analyze_sync(audio, motion, FPS, window_sec=4.0, step_sec=2.0)
    assert rep.verdict == Verdict.DRIFT
    assert rep.lag_slope_ms_per_s > 0


def test_verdict_inconclusive_silent_audio():
    motion = burst_train(300, seed=6)
    rep = analyze_sync(np.zeros(300), motion, FPS, window_sec=4.0, step_sec=2.0)
    assert rep.verdict == Verdict.INCONCLUSIVE


def test_verdict_inconclusive_static_scene():
    audio = burst_train(300, seed=7)
    rep = analyze_sync(audio, np.zeros(300), FPS, window_sec=4.0, step_sec=2.0)
    assert rep.verdict == Verdict.INCONCLUSIVE


def test_report_serializes():
    motion = burst_train(300, seed=8)
    rep = analyze_sync(motion, motion, FPS, window_sec=4.0, step_sec=2.0)
    d = rep.to_dict()
    assert d["verdict"] == "IN_SYNC"
    assert len(d["windows"]) == d["n_windows"]
    assert set(d) >= {"verdict", "fps", "windows", "suspect_segments", "notes"}
