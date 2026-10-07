"""desync-detect: audio-video desynchronization detector.

A fast, CPU-only triage tool that flags A/V lip-sync drift in video files
by correlating audio and visual-motion energy envelopes. No GPU, no deep
models — an envelope-correlation signal for analysts.
"""

__version__ = "0.1.0"
__author__ = "Habib Ur Rehman"

from desyncdetect.features import extract_envelopes, load_audio_envelope, load_motion_envelope
from desyncdetect.sync import analyze_sync, Verdict

__all__ = [
    "__version__",
    "extract_envelopes",
    "load_audio_envelope",
    "load_motion_envelope",
    "analyze_sync",
    "Verdict",
]
