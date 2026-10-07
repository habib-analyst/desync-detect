"""Envelope extraction: audio energy + visual motion energy at video frame rate.

Offline, CPU-only, light deps (numpy, scipy, opencv-python-headless).

Audio input is a WAV file (16-bit PCM, mono or stereo) read with the
standard-library ``wave`` module — OpenCV cannot decode audio tracks, so
for real-world videos extract the audio track first, e.g.::

    ffmpeg -i video.mp4 -ar 16000 -ac 1 audio.wav

``desync-detect analyze`` then takes ``video.mp4 --audio audio.wav``.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Envelopes:
    """Audio and motion energy envelopes sampled at the video frame rate."""

    audio: np.ndarray      # audio energy per video frame
    motion: np.ndarray     # motion energy per video frame
    fps: float             # video frame rate
    n_frames: int


# ---------------------------------------------------------------------------
# audio
# ---------------------------------------------------------------------------

def read_wav_mono(path: str) -> tuple[np.ndarray, int]:
    """Read a WAV file, return (mono float32 samples in [-1, 1], sample_rate)."""
    with wave.open(path, "rb") as w:
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        sr = w.getframerate()
        n_frames = w.getnframes()
        raw = w.readframes(n_frames)
    if sampwidth == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported WAV sample width: {sampwidth} bytes")
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return data.astype(np.float32), sr


def audio_envelope_at_fps(samples: np.ndarray, sample_rate: int, fps: float,
                          n_frames: int, smooth: int = 3) -> np.ndarray:
    """Per-video-frame RMS energy of the audio signal.

    Frame ``i`` covers audio time ``[i/fps, (i+1)/fps)``; its energy is the
    RMS of the samples inside that slice. Lightly smoothed with a moving
    average of width ``smooth`` frames.
    """
    env = np.zeros(n_frames, dtype=np.float64)
    for i in range(n_frames):
        a = int(round(i / fps * sample_rate))
        b = int(round((i + 1) / fps * sample_rate))
        b = max(b, a + 1)
        if a >= len(samples):
            break
        seg = samples[a:min(b, len(samples))]
        env[i] = float(np.sqrt(np.mean(seg ** 2)))
    if smooth > 1:
        kernel = np.ones(smooth) / smooth
        env = np.convolve(env, kernel, mode="same")
    return env


def load_audio_envelope(wav_path: str, fps: float, n_frames: int) -> np.ndarray:
    """Load a WAV file and return its energy envelope at ``fps``."""
    samples, sr = read_wav_mono(wav_path)
    return audio_envelope_at_fps(samples, sr, fps, n_frames)


# ---------------------------------------------------------------------------
# video / motion
# ---------------------------------------------------------------------------

def video_fps_and_count(path: str) -> tuple[float, int]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps <= 0:
        raise ValueError(f"could not determine FPS for {path}")
    return float(fps), n


def load_motion_envelope(video_path: str, max_frames: int | None = None) -> tuple[np.ndarray, float]:
    """Motion-energy envelope: mean abs frame difference per frame.

    Returns (envelope, fps). ``envelope[0]`` is 0 (no previous frame).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise ValueError(f"could not determine FPS for {video_path}")

    env: list[float] = []
    prev: np.ndarray | None = None
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        if prev is None:
            env.append(0.0)
        else:
            env.append(float(np.mean(np.abs(gray - prev))))
        prev = gray
        idx += 1
        if max_frames is not None and idx >= max_frames:
            break
    cap.release()
    if not env:
        raise ValueError(f"no frames decoded from {video_path}")
    return np.asarray(env, dtype=np.float64), float(fps)


def extract_envelopes(video_path: str, audio_path: str) -> Envelopes:
    """Extract audio + motion envelopes from a video file and a WAV file."""
    motion, fps = load_motion_envelope(video_path)
    audio = load_audio_envelope(audio_path, fps, len(motion))
    return Envelopes(audio=audio, motion=motion, fps=fps, n_frames=len(motion))


def save_envelopes_npz(path: str, env: Envelopes) -> None:
    np.savez(path, audio_env=env.audio, motion_env=env.motion, fps=env.fps)


def load_envelopes_npz(path: str) -> Envelopes:
    d = np.load(path)
    audio = np.asarray(d["audio_env"], dtype=np.float64)
    motion = np.asarray(d["motion_env"], dtype=np.float64)
    fps = float(d["fps"])
    n = min(len(audio), len(motion))
    return Envelopes(audio=audio[:n], motion=motion[:n], fps=fps, n_frames=n)
