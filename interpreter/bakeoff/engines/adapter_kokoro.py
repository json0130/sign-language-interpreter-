"""Kokoro-82M: neutral baseline. It has NO emotion control, so every emotion gets the same voice and speed."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import EMOTIONS, run_engine  # noqa: E402

# emotion -> synthesis settings. Kokoro cannot express emotion; this dict is deliberately identical per emotion.
EMOTION_MAP = {e: {"voice": "bf_emma", "speed": 1.0} for e in EMOTIONS}


def main():
    import numpy as np
    import torch
    from kokoro import KPipeline
    pipe = KPipeline(lang_code="b")   # British English

    def synth(text, emotion):
        torch.manual_seed(0)
        cfg = EMOTION_MAP[emotion]
        chunks = [r.audio for r in pipe(text, voice=cfg["voice"], speed=cfg["speed"]) if r.audio is not None]
        return np.concatenate([c.detach().cpu().numpy() for c in chunks]), 24000

    run_engine("kokoro", synth, {"model": "hexgrad/Kokoro-82M", "device": "cuda" if torch.cuda.is_available() else "cpu"})


if __name__ == "__main__":
    main()
