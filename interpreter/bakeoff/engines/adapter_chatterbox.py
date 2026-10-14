"""Chatterbox Multilingual (Resemble AI): emotion is ONE intensity knob (exaggeration), not a label."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import REFERENCE, run_engine  # noqa: E402

# emotion -> (exaggeration, cfg_weight). exaggeration 0.5 is the default; higher = more dramatic delivery and
# lower cfg_weight = slower, more deliberate pacing (per the Chatterbox README). These are guesses to tune by ear.
# The model has no sad/angry label, so valence can only come from the words and the reference voice.
EMOTION_MAP = {
    "neutral": (0.5, 0.5),
    "sad": (0.7, 0.3),
    "excited": (1.2, 0.3),
    "angry": (1.5, 0.5),
}


def main():
    import torch
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")

    def synth(text, emotion):
        torch.manual_seed(0)
        exag, cfg = EMOTION_MAP[emotion]
        return model.generate(text, language_id="en", audio_prompt_path=str(REFERENCE), exaggeration=exag, cfg_weight=cfg), model.sr

    run_engine("chatterbox", synth, {"model": "ResembleAI/chatterbox (Multilingual V3), language_id=en", "device": "cuda"})


if __name__ == "__main__":
    main()
