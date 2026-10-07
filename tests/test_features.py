"""Envelope extraction tests: synthetic signals, no network."""

import numpy as np

from desyncdetect.features import (
    audio_envelope_at_fps,
    load_audio_envelope,
    load_motion_envelope,
)
from desyncdetect.synth import render_audio, render_video


def _bursts(n_frames, fps, period_frames=25, width=3, amp=1.0, shift=0):
    """Square-wave burst train (like a beep/flash train)."""
    x = np.zeros(n_frames)
    for start in range(10 + shift, n_frames - width, period_frames):
        x[start:start + width] = amp
    return x


def test_audio_envelope_peaks_follow_beeps(tmp_path):
    # audio beeps every 1 s at 16 kHz; envelope at 25 fps must peak near them
    wav = str(tmp_path / "a.wav")
    render_audio(wav, beep_times=[0.5, 1.5, 2.5], duration=3.0)
    env = load_audio_envelope(wav, fps=25.0, n_frames=75)
    assert len(env) == 75
    for t in (0.5, 1.5, 2.5):
        i = int(round(t * 25))
        local = env[max(0, i - 2): i + 3]
        assert env[i] >= 0.5 * local.max()  # frame at beep time is energetic
    # silence between beeps is quiet relative to beeps
    assert env[int(1.0 * 25)] < 0.3 * env.max()


def test_motion_envelope_detects_flashes(tmp_path):
    vid = str(tmp_path / "v.mp4")
    render_video(vid, flash_times=[0.5, 1.5], duration=2.0, fps=25.0)
    env, fps = load_motion_envelope(vid)
    assert fps == 25.0
    assert len(env) == 50
    # frames where the square appears/disappears carry the motion energy
    assert env.max() > 10 * np.median(env) or env.max() > 0.01
    peak_frame = int(np.argmax(env))
    assert abs(peak_frame / 25.0 - 0.5) < 0.2 or abs(peak_frame / 25.0 - 1.5) < 0.2


def test_audio_envelope_resampling_shapes():
    sr, fps, n = 16000, 25.0, 50
    rng = np.random.default_rng(0)
    samples = rng.standard_normal(sr * 2).astype(np.float32)
    env = audio_envelope_at_fps(samples, sr, fps, n)
    assert env.shape == (n,)
    assert np.all(env >= 0)


def test_flat_video_gives_flat_motion_envelope(tmp_path):
    vid = str(tmp_path / "static.mp4")
    render_video(vid, flash_times=[], duration=1.0, fps=25.0)
    env, _ = load_motion_envelope(vid)
    assert np.std(env) < 1e-6  # no flashes -> (near-)zero motion everywhere
