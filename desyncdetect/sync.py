"""A/V sync analysis: normalized cross-correlation over sliding windows.

Sign convention: ``lag_ms > 0`` means the **audio lags behind the video**
(audio arrives later than the matching visual motion); ``lag_ms < 0``
means audio leads video.

Verdicts:
    IN_SYNC       median |lag| within tolerance
    DESYNCED      sustained |lag| above tolerance
    DRIFT         lag grows (in magnitude) over time — e.g. clock drift
    INCONCLUSIVE  not enough signal (silent audio / static scene)

Quasi-periodic signals produce alias correlation peaks one period apart;
per-window lags are disambiguated by a greedy continuity pass
(``unwrap_lag_sequence``) before the verdict is computed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy.signal import correlate, correlation_lags
from scipy.stats import theilslopes


class Verdict(str, Enum):
    IN_SYNC = "IN_SYNC"
    DESYNCED = "DESYNCED"
    DRIFT = "DRIFT"
    INCONCLUSIVE = "INCONCLUSIVE"


# --- tunables (documented, overridable) ------------------------------------
LAG_TOLERANCE_MS = 150.0   # |lag| below this counts as "in sync"
MIN_CONFIDENCE = 0.35      # per-window confidence floor for a valid window
DRIFT_MIN_CONFIDENCE = 0.20  # lower bar for drift-trend windows: a consistent
                             # lag trend across many windows is itself significant,
                             # so single-window confidence can be relaxed
MIN_DRIFT_WINDOWS = 4        # windows needed (at the drift bar) to test a trend
MAX_LAG_MS = 2000.0        # correlation search range
DRIFT_SLOPE_MS_PER_S = 30.0  # |d(lag)/dt| above this => drift (triage-leaning;
                             # the 250 ms range gate below does the heavy
                             # lifting against noise)
DRIFT_RANGE_MS = 250.0       # total lag swing needed to call drift
PEAK_NEIGHBORHOOD = 5        # frames excluded around peak when estimating floor


@dataclass
class WindowResult:
    t_start: float
    t_end: float
    lag_ms: float
    confidence: float
    valid: bool
    fps: float = 25.0
    # top-k (lag_ms, peak) candidates + full |xcorr| curve for unwrapping;
    # internal detail, not serialized into the JSON report.
    candidates: list[tuple[float, float]] = field(default_factory=list)
    xcorr: tuple | None = None


@dataclass
class SyncReport:
    verdict: Verdict
    fps: float
    window_sec: float
    n_windows: int
    n_valid: int
    median_lag_ms: float
    lag_slope_ms_per_s: float
    windows: list[WindowResult] = field(default_factory=list)
    suspect_segments: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "fps": self.fps,
            "window_sec": self.window_sec,
            "n_windows": self.n_windows,
            "n_valid_windows": self.n_valid,
            "median_lag_ms": round(self.median_lag_ms, 1),
            "lag_slope_ms_per_s": round(self.lag_slope_ms_per_s, 1),
            "windows": [
                {
                    "t_start_s": round(w.t_start, 2),
                    "t_end_s": round(w.t_end, 2),
                    "lag_ms": round(w.lag_ms, 1),
                    "confidence": round(w.confidence, 3),
                    "valid": w.valid,
                }
                for w in self.windows
            ],
            "suspect_segments": self.suspect_segments,
            "notes": self.notes,
        }


def _znorm(x: np.ndarray) -> tuple[np.ndarray, float]:
    x = np.asarray(x, dtype=np.float64)
    std = float(np.std(x))
    if std < 1e-12:
        return np.zeros_like(x), 0.0
    return (x - x.mean()) / std, std


def _xcorr_mag(audio: np.ndarray, motion: np.ndarray, fps: float,
               max_lag_ms: float):
    """Normalized |cross-correlation| magnitude + lag axis (frames→ms later)."""
    a, a_std = _znorm(audio)
    m, m_std = _znorm(motion)
    n = len(a)
    lags_frames = correlation_lags(n, len(m), mode="full")
    in_range = np.abs(lags_frames) <= int(round(max_lag_ms / 1000.0 * fps))
    if a_std < 1e-12 or m_std < 1e-12 or not np.any(in_range):
        return None
    corr = correlate(a, m, mode="full") / n  # in [-1, 1] after znorm
    return np.abs(corr), lags_frames, in_range


def _confidence(mag: np.ndarray, in_range: np.ndarray, peak_pos: int) -> float:
    """Peak prominence above the correlation floor, in [0, 1]."""
    sel_idx = np.flatnonzero(in_range)
    peak = float(mag[peak_pos])
    lo = max(sel_idx[0], peak_pos - PEAK_NEIGHBORHOOD)
    hi = min(sel_idx[-1] + 1, peak_pos + PEAK_NEIGHBORHOOD + 1)
    floor_mask = np.ones_like(mag, dtype=bool)
    floor_mask[lo:hi] = False
    floor_mask &= in_range
    floor = float(np.mean(mag[floor_mask])) if np.any(floor_mask) else 0.0
    return float(np.clip((peak - floor) / (1.0 - floor + 1e-9), 0.0, 1.0))


def lag_candidates(audio: np.ndarray, motion: np.ndarray, fps: float,
                   max_lag_ms: float = MAX_LAG_MS, k: int = 3,
                   ) -> list[tuple[float, float]]:
    """Top-``k`` (lag_ms, peak) candidates by |normalized cross-correlation|.

    Quasi-periodic signals (speech rhythm, repeated motion) produce alias
    peaks one period apart; returning several candidates lets the caller
    disambiguate via continuity across windows.
    """
    out = _xcorr_mag(audio, motion, fps, max_lag_ms)
    if out is None:
        return []
    mag, lags_frames, in_range = out
    sel_idx = np.flatnonzero(in_range)
    cands: list[tuple[float, float]] = []
    masked = mag.copy()
    masked[~in_range] = -1.0
    for _ in range(k):
        pos = int(np.argmax(masked))
        if masked[pos] < 0:
            break
        cands.append((float(lags_frames[pos]) / fps * 1000.0, float(masked[pos])))
        lo = max(0, pos - PEAK_NEIGHBORHOOD)
        masked[lo:pos + PEAK_NEIGHBORHOOD + 1] = -1.0
    return cands


def cross_correlation_lag(audio: np.ndarray, motion: np.ndarray, fps: float,
                          max_lag_ms: float = MAX_LAG_MS) -> tuple[float, float]:
    """Lag (ms) between audio and motion envelopes + confidence in [0, 1].

    Normalized cross-correlation; the peak location gives the lag, the peak
    prominence above the correlation floor gives the confidence.
    """
    out = _xcorr_mag(audio, motion, fps, max_lag_ms)
    if out is None:
        return 0.0, 0.0
    mag, lags_frames, in_range = out
    sel_idx = np.flatnonzero(in_range)
    peak_pos = int(sel_idx[np.argmax(mag[in_range])])
    confidence = _confidence(mag, in_range, peak_pos)
    lag_ms = float(lags_frames[peak_pos]) / fps * 1000.0
    return lag_ms, confidence


def unwrap_lag_sequence(windows: list["WindowResult"],
                        peak_ratio: float = 0.60) -> None:
    """Fix alias jumps: greedy continuity pass over per-window candidates.

    For each window after the first, among candidates whose peak is within
    ``peak_ratio`` of the best peak, keep the lag closest to the previous
    window's lag. The ratio is deliberately permissive: quasi-periodic
    content can make an alias peak *stronger* than the true peak, so
    continuity across windows is the more reliable disambiguator.
    Mutates ``windows`` in place (lag_ms and confidence are recomputed for
    the chosen candidate where curves are available).
    """
    for i in range(1, len(windows)):
        prev_lag = windows[i - 1].lag_ms
        cands = windows[i].candidates
        if not cands:
            continue
        best_peak = cands[0][1]
        near = [c for c in cands if c[1] >= peak_ratio * best_peak]
        chosen = min(near, key=lambda c: abs(c[0] - prev_lag))
        windows[i].lag_ms = chosen[0]
        if windows[i].xcorr is not None:
            mag, lags_frames, in_range = windows[i].xcorr
            # nearest correlation bin to the chosen lag
            target = chosen[0] / 1000.0 * windows[i].fps
            sel_idx = np.flatnonzero(in_range)
            pos = int(sel_idx[np.argmin(np.abs(lags_frames[in_range] - target))])
            windows[i].confidence = _confidence(mag, in_range, pos)


def analyze_sync(audio_env: np.ndarray, motion_env: np.ndarray, fps: float,
                 window_sec: float = 5.0, step_sec: float | None = None,
                 lag_tolerance_ms: float = LAG_TOLERANCE_MS,
                 min_confidence: float = MIN_CONFIDENCE) -> SyncReport:
    """Sliding-window A/V sync analysis → per-window lags + global verdict."""
    n = min(len(audio_env), len(motion_env))
    audio_env = np.asarray(audio_env, dtype=np.float64)[:n]
    motion_env = np.asarray(motion_env, dtype=np.float64)[:n]
    step_sec = window_sec / 2.0 if step_sec is None else step_sec
    win = max(2, int(round(window_sec * fps)))
    step = max(1, int(round(step_sec * fps)))

    notes: list[str] = []
    if float(np.std(audio_env)) < 1e-9:
        notes.append("audio envelope is flat (silent audio?) — windows will be invalid")
    if float(np.std(motion_env)) < 1e-9:
        notes.append("motion envelope is flat (static scene?) — windows will be invalid")

    windows: list[WindowResult] = []
    start = 0
    while start + win <= n:
        a = audio_env[start:start + win]
        m = motion_env[start:start + win]
        lag_ms, conf = cross_correlation_lag(a, m, fps)
        valid = conf >= min_confidence
        w = WindowResult(
            t_start=start / fps, t_end=(start + win) / fps,
            lag_ms=lag_ms, confidence=conf, valid=valid, fps=fps,
        )
        # keep candidates + curve so alias jumps can be unwrapped below
        w.candidates = lag_candidates(a, m, fps)
        xc = _xcorr_mag(a, m, fps, MAX_LAG_MS)
        w.xcorr = xc
        windows.append(w)
        start += step

    # disambiguate quasi-periodic alias peaks via continuity across windows
    unwrap_lag_sequence(windows)
    for w in windows:
        w.valid = w.confidence >= min_confidence

    valid_w = [w for w in windows if w.valid]
    if len(valid_w) < 2:
        return SyncReport(
            verdict=Verdict.INCONCLUSIVE, fps=fps, window_sec=window_sec,
            n_windows=len(windows), n_valid=len(valid_w),
            median_lag_ms=0.0, lag_slope_ms_per_s=0.0,
            windows=windows, suspect_segments=[],
            notes=notes + ["fewer than 2 valid windows — cannot judge sync"],
        )

    lags = np.array([w.lag_ms for w in valid_w])
    centers = np.array([(w.t_start + w.t_end) / 2.0 for w in valid_w])
    median_lag = float(np.median(np.abs(lags)))

    # drift: robust (Theil-Sen) slope of lag vs time over windows clearing
    # the (lower) drift confidence bar. Within-window drift smears the
    # correlation peak, so single-window confidence is relaxed here; the
    # trend across windows provides the statistical significance instead.
    # Theil-Sen resists isolated alias jumps (quasi-periodic content can
    # throw one window off by a whole signal period).
    drift_w = [w for w in windows if w.confidence >= DRIFT_MIN_CONFIDENCE]
    if len(drift_w) >= MIN_DRIFT_WINDOWS:
        d_lags = np.array([w.lag_ms for w in drift_w])
        d_centers = np.array([(w.t_start + w.t_end) / 2.0 for w in drift_w])
        slope = float(theilslopes(d_lags, d_centers)[0])
        lag_range = float(np.max(d_lags) - np.min(d_lags))
    else:
        slope, lag_range = 0.0, 0.0

    bad = [w for w in valid_w if abs(w.lag_ms) > lag_tolerance_ms]
    bad_frac = len(bad) / len(valid_w)

    if abs(slope) > DRIFT_SLOPE_MS_PER_S and lag_range > DRIFT_RANGE_MS:
        verdict = Verdict.DRIFT
        notes.append(f"lag drifts {slope:+.0f} ms/s over the clip")
    elif bad_frac >= 0.5 and median_lag > lag_tolerance_ms:
        verdict = Verdict.DESYNCED
        notes.append(f"{len(bad)}/{len(valid_w)} windows exceed ±{lag_tolerance_ms:.0f} ms")
    else:
        verdict = Verdict.IN_SYNC
        notes.append(f"median |lag| {median_lag:.0f} ms within ±{lag_tolerance_ms:.0f} ms")

    # suspect segments: contiguous runs of bad valid windows
    segments: list[dict] = []
    run: list[WindowResult] = []
    for w in valid_w:
        if abs(w.lag_ms) > lag_tolerance_ms:
            run.append(w)
        else:
            if run:
                segments.append(_segment(run))
                run = []
    if run:
        segments.append(_segment(run))

    return SyncReport(
        verdict=verdict, fps=fps, window_sec=window_sec,
        n_windows=len(windows), n_valid=len(valid_w),
        median_lag_ms=median_lag, lag_slope_ms_per_s=slope,
        windows=windows, suspect_segments=segments, notes=notes,
    )


def _segment(run: list[WindowResult]) -> dict:
    return {
        "t_start_s": round(run[0].t_start, 2),
        "t_end_s": round(run[-1].t_end, 2),
        "median_lag_ms": round(float(np.median([w.lag_ms for w in run])), 1),
        "n_windows": len(run),
    }
