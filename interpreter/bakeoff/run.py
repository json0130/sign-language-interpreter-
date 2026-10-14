"""TTS emotion bake-off driver (throwaway evaluation). Standard library only.

    python3 interpreter/bakeoff/run.py [--engines a,b] [--skip-render] [--skip-ser] [--device cuda]

1. render: each engine's adapter runs inside its own venv (venvs/<name>/), writing out/<engine>/*.wav
2. normalise: every clip is scaled to the same speech-active RMS so loudness does not bias listening
   (originals are kept in out/<engine>/raw/)
3. ser: venvs/ser runs an off-the-shelf speech-emotion recogniser over every clip (a rough proxy)
4. summary: out/results.json plus a markdown table on stdout
"""
import argparse
import array
import json
import os
import shutil
import statistics
import subprocess
import wave
from pathlib import Path

from common import EMOTIONS, OUT, REFERENCE, ROOT, SENTENCES

HERE = Path(__file__).resolve().parent
# engine name -> venv directory name under venvs/ (the adapter is engines/adapter_<name>.py)
ENGINES = {"kokoro": "kokoro", "parler": "parler", "chatterbox": "chatterbox", "cosyvoice": "cosyvoice",
           "indextts": "indextts"}
TARGET_RMS, PEAK_LIMIT = 0.08, 0.95
# SER has no "excited" class, so happy or surprised counts as a match for it. A rough mapping, stated in the report.
ACCEPT = {"neutral": {"neutral"}, "sad": {"sad"}, "excited": {"happy", "surprised"}, "angry": {"angry"}}


# ---------------------------------------------------------------- 1. render
def render(name: str) -> dict:
    py = HERE / "venvs" / ENGINES[name] / "bin" / "python"
    if not py.exists():
        return {"status": "no venv", "error": f"{py} does not exist"}
    (OUT / name).mkdir(parents=True, exist_ok=True)
    shutil.rmtree(OUT / name / "raw", ignore_errors=True)    # a fresh render replaces old originals
    for old in (OUT / name).glob("*.wav"):
        old.unlink()
    env = {**os.environ, "PYTHONPATH": str(ROOT), "HF_HOME": str(HERE / "models" / "hf")}
    with open(OUT / name / "run.log", "w") as log:
        rc = subprocess.run([str(py), str(HERE / "engines" / f"adapter_{name}.py")], env=env, stdout=log,
                            stderr=subprocess.STDOUT).returncode
    if rc:
        tail = (OUT / name / "run.log").read_text(errors="replace").strip().splitlines()[-3:]
        return {"status": "failed", "error": " | ".join(tail)[:400]}
    return {"status": "ok"}


# ---------------------------------------------------------------- 2. normalise
def _read(path: Path):
    with wave.open(str(path), "rb") as w:
        s = array.array("h")
        s.frombytes(w.readframes(w.getnframes()))
        return s, w.getframerate()


def active_rms(samples, sr: int) -> float:
    """RMS over 20 ms frames that are within ~20 dB of the loudest frame (speech, not silence)."""
    n = max(1, sr // 50)
    energy = [sum(x * x for x in samples[i:i + n]) / len(samples[i:i + n]) for i in range(0, len(samples), n)]
    gate = max(energy) * 0.01
    kept = [e for e in energy if e >= gate]
    return (sum(kept) / len(kept)) ** 0.5 / 32768 if kept else 0.0


def normalise(engine_dir: Path) -> None:
    raw = engine_dir / "raw"
    raw.mkdir(exist_ok=True)
    for wav in engine_dir.glob("*.wav"):
        if not (raw / wav.name).exists():      # already in raw/ means this is an earlier normalised copy
            shutil.move(wav, raw / wav.name)
    for wav in sorted(raw.glob("*.wav")):
        s, sr = _read(wav)
        rms, peak = active_rms(s, sr), max((abs(x) for x in s), default=0) / 32768
        gain = TARGET_RMS / rms if rms else 1.0
        if peak * gain > PEAK_LIMIT:
            gain = PEAK_LIMIT / peak          # rare: keep headroom, clip nothing
        out = array.array("h", (max(-32768, min(32767, round(x * gain))) for x in s))
        with wave.open(str(engine_dir / wav.name), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(out.tobytes())


# ---------------------------------------------------------------- 3. SER
def run_ser(device: str) -> None:
    py = HERE / "venvs" / "ser" / "bin" / "python"
    env = {**os.environ, "HF_HOME": str(HERE / "models" / "hf"), "MODELSCOPE_CACHE": str(HERE / "models" / "ms")}
    subprocess.run([str(py), str(HERE / "ser.py"), "--device", device], env=env, check=True)


# ---------------------------------------------------------------- 4. summary
def summarise(status: dict) -> dict:
    ser = json.loads((OUT / "ser.json").read_text())["results"] if (OUT / "ser.json").exists() else {}
    rows = {}
    for name in ENGINES:
        d = OUT / name
        timings = json.loads((d / "timings.json").read_text()) if (d / "timings.json").exists() else []
        meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
        ok = [t for t in timings if "error" not in t]
        rtf, rtf_max = {}, {}
        for sid in SENTENCES:   # median over the 4 emotions: robust to one-off slow calls (Parler has a few)
            v = sorted(t["rtf"] for t in ok if t["file"].startswith(sid + "__") and t.get("rtf"))
            rtf[sid] = round(statistics.median(v), 2) if v else None
            rtf_max[sid] = round(v[-1], 2) if v else None
        confusion, match, n = {e: {} for e in EMOTIONS}, 0, 0
        for sid in SENTENCES:
            for emo in EMOTIONS:
                r = ser.get(name, {}).get(f"{sid}__{emo}.wav")
                if r:
                    n += 1
                    confusion[emo][r["label"]] = confusion[emo].get(r["label"], 0) + 1
                    match += r["label"] in ACCEPT[emo]
        fallback = "ok" if ok and len(ok) == len(timings) else "not run"   # e.g. an adapter run directly, not via render()
        rows[name] = {"status": status.get(name, {}).get("status", fallback), "error": status.get(name, {}).get("error"),
                      "rendered": f"{len(ok)}/{len(timings)}", "rtf_median_by_sentence": rtf, "rtf_max_by_sentence": rtf_max, "meta": meta,
                      "ser_match": f"{match}/{n}" if n else None, "ser_confusion": confusion}
    (OUT / "results.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("| engine | status | rendered | median RTF (max) s1 / s2 / s3 | peak VRAM GB | SER match (of 12) |\n|---|---|---|---|---|---|")
    for name, r in rows.items():
        rtf = " / ".join(f"{r['rtf_median_by_sentence'][s]} (max {r['rtf_max_by_sentence'][s]})" for s in SENTENCES)
        print(f"| {name} | {r['status']} | {r['rendered']} | {rtf} | {r['meta'].get('peak_vram_gb', '-')} | {r['ser_match'] or '-'} |")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engines", default=",".join(ENGINES))
    ap.add_argument("--skip-render", action="store_true")
    ap.add_argument("--skip-ser", action="store_true")
    ap.add_argument("--device", default="cpu", help="device for the SER model")
    args = ap.parse_args()
    names = [n for n in args.engines.split(",") if n]
    status = json.loads((OUT / "status.json").read_text()) if (OUT / "status.json").exists() else {}
    for name in names:
        if not REFERENCE.exists() and (OUT / "kokoro" / "raw" / "s2__neutral.wav").exists():
            REFERENCE.parent.mkdir(exist_ok=True)
            shutil.copy(OUT / "kokoro" / "raw" / "s2__neutral.wav", REFERENCE)   # Kokoro must be rendered first
        if not args.skip_render:
            print(f"== rendering {name}", flush=True)
            status[name] = render(name)
            print("  ", status[name])
        if (OUT / name).exists() and list((OUT / name).glob("*.wav")) + list((OUT / name / "raw").glob("*.wav")):
            normalise(OUT / name)
    OUT.mkdir(exist_ok=True)
    (OUT / "status.json").write_text(json.dumps(status, indent=1))
    if not args.skip_ser:
        run_ser(args.device)
    summarise(status)


if __name__ == "__main__":
    main()
