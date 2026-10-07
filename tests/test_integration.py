"""End-to-end tests: synth -> extract -> analyze -> CLI (tiny videos, fast)."""

import json
import subprocess
import sys

from desyncdetect.cli import main as cli_main
from desyncdetect.features import extract_envelopes, load_envelopes_npz, save_envelopes_npz
from desyncdetect.sync import analyze_sync, Verdict
from desyncdetect.synth import synthesize


def _verdicts(tmp_path):
    samples = synthesize(str(tmp_path))
    out = {}
    for name, p in samples.items():
        env = extract_envelopes(p["video"], p["audio"])
        rep = analyze_sync(env.audio, env.motion, env.fps, window_sec=4.0, step_sec=2.0)
        out[name] = rep
    return out


def test_synth_produces_expected_verdicts(tmp_path):
    reps = _verdicts(tmp_path)
    assert reps["clean"].verdict == Verdict.IN_SYNC, reps["clean"].notes
    assert reps["lagged"].verdict == Verdict.DESYNCED, reps["lagged"].notes
    assert reps["drift"].verdict == Verdict.DRIFT, reps["drift"].notes


def test_lagged_sample_reports_planted_lag(tmp_path):
    reps = _verdicts(tmp_path)
    lag = reps["lagged"].median_lag_ms
    assert 300 < lag < 500, f"planted 400 ms lag, recovered {lag} ms"


def test_npz_roundtrip_and_from_npz_mode(tmp_path):
    samples = synthesize(str(tmp_path))
    p = samples["clean"]
    env = extract_envelopes(p["video"], p["audio"])
    npz = str(tmp_path / "env.npz")
    save_envelopes_npz(npz, env)
    env2 = load_envelopes_npz(npz)
    assert env2.fps == env.fps == 25.0
    rep = analyze_sync(env2.audio, env2.motion, env2.fps, window_sec=4.0, step_sec=2.0)
    assert rep.verdict == Verdict.IN_SYNC
    # CLI --from-npz path
    rc = cli_main(["analyze", "--from-npz", npz, "--window", "4", "--step", "2"])
    assert rc == 0


def test_cli_analyze_writes_report_json(tmp_path):
    samples = synthesize(str(tmp_path))
    p = samples["lagged"]
    report = str(tmp_path / "report.json")
    rc = cli_main(["analyze", p["video"], "--audio", p["audio"],
                   "--window", "4", "--step", "2", "--out", report])
    assert rc == 0
    d = json.loads(open(report).read())
    assert d["verdict"] == "DESYNCED"
    assert d["n_valid_windows"] >= 2


def test_cli_demo_smoke(capsys):
    rc = cli_main(["demo"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "clean" in out and "IN_SYNC" in out
    assert "lagged" in out and "DESYNCED" in out
    assert "drift" in out and "DRIFT" in out


def test_module_entrypoint():
    r = subprocess.run([sys.executable, "-m", "desyncdetect", "demo"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "demo passed" in r.stdout
