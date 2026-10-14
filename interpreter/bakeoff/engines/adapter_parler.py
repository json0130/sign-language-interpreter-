"""Parler-TTS Mini v1.1: style is set by a free-text voice description."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import run_engine  # noqa: E402

REPO = "parler-tts/parler-tts-mini-v1.1"
# emotion -> voice description. The same named speaker ("Jon") is kept in every description for a stable voice.
_BASE = "with very clear audio and no background noise."
EMOTION_MAP = {
    "neutral": f"Jon's voice is calm and neutral, at a moderate speed and pitch, {_BASE}",
    "sad": f"Jon's voice is sad and subdued, slow and low-pitched, quiet and heavy with sorrow, {_BASE}",
    "excited": f"Jon speaks in an excited, animated and very expressive way, fast with high energy and a high pitch, {_BASE}",
    "angry": f"Jon speaks in an angry, harsh and forceful voice, loud and fast, with intense emotion, {_BASE}",
}


def main():
    import torch
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    model = ParlerTTSForConditionalGeneration.from_pretrained(REPO).to("cuda")
    tok = AutoTokenizer.from_pretrained(REPO)

    def synth(text, emotion):
        torch.manual_seed(0)
        desc = tok(EMOTION_MAP[emotion], return_tensors="pt").input_ids.to("cuda")
        prompt = tok(text, return_tensors="pt").input_ids.to("cuda")
        audio = model.generate(input_ids=desc, prompt_input_ids=prompt)
        return audio.cpu().float().numpy().squeeze(), model.config.sampling_rate

    run_engine("parler", synth, {"model": REPO, "device": "cuda"})


if __name__ == "__main__":
    main()
