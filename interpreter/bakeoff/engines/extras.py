"""Render the other emotions of our contract on one engine, sentence 2 only, to see how flexible the emotion input is.
Run in that engine's venv:  PYTHONPATH=../.. venvs/<engine>/bin/python engines/extras.py indextts|cosyvoice
Writes out/<engine>_extra/s2__<emotion>.wav (raw; normalise with run.normalise)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, REFERENCE, SENTENCES, save_wav  # noqa: E402

TEXT = SENTENCES["s2"]
EXTRA = ("happy", "calm", "apologetic", "anxious", "surprised", "confused")


def indextts(out: Path) -> None:
    import numpy as np
    import torch
    import adapter_indextts as ai
    sys.path.insert(0, str(ai.REPO))
    from indextts.infer_v2_5 import IndexTTS2
    tts = IndexTTS2(cfg_path=str(ai.MODEL_DIR / "config.yaml"), model_dir=str(ai.MODEL_DIR), use_bf16=True)
    runs = [(e, ai.EMOTION_MAP[e], ai.ALPHA_BY_EMOTION[e], 1.0) for e in EXTRA]
    runs.append(("calm_slow", ai.EMOTION_MAP["calm"], ai.ALPHA_BY_EMOTION["calm"], 1.15))   # pace lever: >1 is slower
    for name, vec, alpha, dur in runs:
        torch.manual_seed(0)
        sr, w = tts.infer(spk_audio_prompt=str(REFERENCE), text=TEXT, output_path=None, lang="EN", emo_vector=vec,
                          emo_alpha=alpha, duration_factor=dur, use_random=False)
        print(name, round(save_wav(out / f"s2__{name}.wav", np.asarray(w, dtype="float32").reshape(-1) / 32768.0, sr), 2), flush=True)


def cosyvoice(out: Path) -> None:
    import torch
    import adapter_cosyvoice as ac
    sys.path[:0] = [str(ac.REPO), str(ac.REPO / "third_party" / "Matcha-TTS")]
    from cosyvoice.cli.cosyvoice import AutoModel
    cv = AutoModel(model_dir=str(ac.MODEL_DIR))
    for name, instr in ac.EXTRA_INSTRUCTIONS.items():
        torch.manual_seed(0)
        parts = [o["tts_speech"] for o in cv.inference_instruct2(
            TEXT, f"{ac._A} {instr}<|endofprompt|>", str(REFERENCE), stream=False)]
        print(name, round(save_wav(out / f"s2__{name}.wav", torch.cat(parts, dim=1), cv.sample_rate), 2), flush=True)


if __name__ == "__main__":
    engine = sys.argv[1]
    out = OUT / f"{engine}_extra"
    out.mkdir(parents=True, exist_ok=True)
    {"indextts": indextts, "cosyvoice": cosyvoice}[engine](out)
