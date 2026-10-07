"""Command-line interface for desync-detect."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

from desyncdetect import __version__
from desyncdetect.features import extract_envelopes, load_envelopes_npz, save_envelopes_npz
from desyncdetect.sync import analyze_sync, Verdict
from desyncdetect.synth import synthesize


def cmd_analyze(args: argparse.Namespace) -> int:
    if args.from_npz:
        env = load_envelopes_npz(args.from_npz)
    else:
        if not args.video:
            print("error: provide VIDEO or --from-npz", file=sys.stderr)
            return 2
        if not args.audio:
            print("error: --audio AUDIO.wav is required (OpenCV cannot decode "
                  "audio tracks; extract it first, e.g. "
                  "`ffmpeg -i video.mp4 -ar 16000 -ac 1 audio.wav`)",
                  file=sys.stderr)
            return 2
        env = extract_envelopes(args.video, args.audio)

    report = analyze_sync(env.audio, env.motion, env.fps,
                          window_sec=args.window, step_sec=args.step)
    d = report.to_dict()
    src = args.from_npz or args.video
    d["source"] = src

    if args.out:
        with open(args.out, "w") as f:
            json.dump(d, f, indent=2)
        print(f"report written to {args.out}")

    print(f"verdict: {report.verdict.value}  "
          f"(median |lag| {report.median_lag_ms:.0f} ms, "
          f"{report.n_valid}/{report.n_windows} valid windows)")
    for w in report.windows:
        flag = "" if w.valid else "  [invalid]"
        print(f"  [{w.t_start:5.1f}s - {w.t_end:5.1f}s] "
              f"lag {w.lag_ms:+7.1f} ms  conf {w.confidence:.2f}{flag}")
    for s in report.suspect_segments:
        print(f"  suspect segment: {s['t_start_s']}s - {s['t_end_s']}s "
              f"(median lag {s['median_lag_ms']:+.0f} ms)")
    for note in report.notes:
        print(f"  note: {note}")
    return 0


def cmd_synth(args: argparse.Namespace) -> int:
    samples = synthesize(args.out, fps=args.fps, lag_ms=args.lag_ms,
                         drift_ms_per_s=args.drift)
    for name, p in samples.items():
        print(f"{name:8s} video={p['video']}  audio={p['audio']}")
    return 0


def _analyze_pair(video: str, audio: str, window: float, step: float):
    env = extract_envelopes(video, audio)
    return analyze_sync(env.audio, env.motion, env.fps,
                        window_sec=window, step_sec=step)


def cmd_demo(_args: argparse.Namespace) -> int:
    """Synthesize 3 samples, analyze them, print the verdict table."""
    print(f"desync-detect {__version__} — demo (synthetic samples, offline)\n")
    with tempfile.TemporaryDirectory(prefix="desync_demo_") as tmp:
        samples = synthesize(tmp)
        results: list[tuple[str, object]] = []
        for name in ("clean", "lagged", "drift"):
            p = samples[name]
            report = _analyze_pair(p["video"], p["audio"], window=4.0, step=2.0)
            results.append((name, report))

        print(f"{'sample':10s} {'verdict':12s} {'median|lag|':>11s} {'slope':>9s}  notes")
        print("-" * 78)
        for name, r in results:
            note = r.notes[-1] if r.notes else ""
            print(f"{name:10s} {r.verdict.value:12s} {r.median_lag_ms:8.0f} ms "
                  f"{r.lag_slope_ms_per_s:+8.0f}  {note}")
        print()

        expected = {"clean": Verdict.IN_SYNC, "lagged": Verdict.DESYNCED,
                    "drift": Verdict.DRIFT}
        ok = True
        for name, r in results:
            want = expected[name]
            status = "OK " if r.verdict == want else "FAIL"
            if r.verdict != want:
                ok = False
            print(f"[{status}] {name}: got {r.verdict.value}, expected {want.value}")
        if not ok:
            print("\ndemo FAILED: unexpected verdict(s)", file=sys.stderr)
            return 1
        print("\ndemo passed: clean→IN_SYNC, lagged→DESYNCED, drift→DRIFT")
        return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="desync-detect",
                                description="Audio-video desynchronization detector (triage signal).")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="analyze a video file")
    a.add_argument("video", nargs="?", help="video file (mp4)")
    a.add_argument("--audio", help="WAV audio track for the video")
    a.add_argument("--from-npz", help="load precomputed envelopes instead of a video")
    a.add_argument("--window", type=float, default=5.0, help="window seconds (default 5)")
    a.add_argument("--step", type=float, default=None, help="step seconds (default window/2)")
    a.add_argument("--out", help="write JSON report to this path")
    a.set_defaults(func=cmd_analyze)

    s = sub.add_parser("synth", help="generate synthetic test samples")
    s.add_argument("--out", default="./samples", help="output directory")
    s.add_argument("--fps", type=float, default=25.0)
    s.add_argument("--lag-ms", type=float, default=400.0, help="constant lag for the lagged sample")
    s.add_argument("--drift", type=float, default=50.0, help="drift rate ms/s for the drift sample")
    s.set_defaults(func=cmd_synth)

    d = sub.add_parser("demo", help="synthesize + analyze 3 samples, print verdict table")
    d.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
