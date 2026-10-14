"""Shared pieces for the bake-off adapters. Standard library only (numpy and torch are imported lazily, and only
inside the candidate venvs). Throwaway evaluation code: nothing outside bakeoff/ imports this."""
import json
import re
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # repo root, so `interpreter.lexicon` imports in any venv
sys.path.insert(0, str(ROOT))
from interpreter.lexicon import LEXICON, lookup      # noqa: E402  (stdlib-only module)

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
# One neutral speaker reference for every engine that clones from a reference clip (Chatterbox, CosyVoice, IndexTTS).
# It is Kokoro's neutral s2 clip (run.py copies it here). The repos' own sample prompts carry emotion of their own:
# the SER labels IndexTTS's voice_01.wav "angry" and CosyVoice's prompt "happy", which leaked into the output.
REFERENCE = HERE / "reference" / "neutral_ref.wav"
SENTENCES = {
    "s1": "My whānau are coming to the hui tomorrow.",
    "s2": "I'm sorry, the meeting has been moved to next week.",
    "s3": "The results are in.",
}
EMOTIONS = ("neutral", "sad", "excited", "angry")


def respelled(text: str) -> str:
    """The text with lexicon.py respellings applied the way local_tts does (lowercase, hyphens kept)."""
    for word in LEXICON:
        text = re.sub(rf"\b{re.escape(word)}\b", lookup(word).lower(), text, flags=re.IGNORECASE)
    return text


def jobs():
    """(file stem, text, emotion). Wording is identical across emotions. The last two are the te reo spelling check."""
    for sid, text in SENTENCES.items():
        for emo in EMOTIONS:
            yield f"{sid}__{emo}", text, emo
    yield "te_reo__raw", SENTENCES["s1"], "neutral"
    yield "te_reo__respelled", respelled(SENTENCES["s1"]), "neutral"


def save_wav(path: Path, samples, sr: int) -> float:
    """Write float samples in [-1, 1] (numpy array or torch tensor, any shape) as mono 16-bit WAV; returns seconds."""
    import numpy as np
    x = samples.detach().cpu().float().numpy() if hasattr(samples, "detach") else np.asarray(samples, dtype="float32")
    x = np.clip(x.reshape(-1), -1.0, 1.0)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes((x * 32767).astype("<i2").tobytes())
    return len(x) / sr


def _sync() -> None:
    if "torch" in sys.modules:
        t = sys.modules["torch"]
        if t.cuda.is_available():
            t.cuda.synchronize()


def run_engine(name: str, synth, meta: dict) -> dict:
    """Render every job with `synth(text, emotion) -> (samples, sample_rate)`.

    Writes out/<name>/<stem>.wav plus timings.json (real-time factor = synthesis seconds / audio seconds)
    and meta.json. A warm-up call is made first and not timed. A failing job is recorded and skipped.
    """
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    synth("Warm up.", "neutral")
    rows = []
    for stem, text, emo in jobs():
        row = {"file": f"{stem}.wav", "text": text, "emotion": emo}
        try:
            _sync()
            t0 = time.perf_counter()
            samples, sr = synth(text, emo)
            _sync()
            row["synth_s"] = round(time.perf_counter() - t0, 3)
            row["audio_s"] = round(save_wav(out / f"{stem}.wav", samples, sr), 3)
            row["rtf"] = round(row["synth_s"] / row["audio_s"], 3) if row["audio_s"] else None
            row["sample_rate"] = sr
        except Exception as e:   # noqa: BLE001  record and continue
            row["error"] = f"{type(e).__name__}: {e}"
        print(row, flush=True)
        rows.append(row)
    (out / "timings.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    if "torch" in sys.modules and sys.modules["torch"].cuda.is_available():
        t = sys.modules["torch"]
        meta = {**meta, "torch": t.__version__, "peak_vram_gb": round(t.cuda.max_memory_allocated() / 1e9, 2)}
    (out / "meta.json").write_text(json.dumps({"python": sys.version.split()[0], **meta}, indent=1), encoding="utf-8")
    ok = [r for r in rows if "error" not in r]
    print(f"{name}: {len(ok)}/{len(rows)} files rendered")
    return {"ok": len(ok), "total": len(rows)}
