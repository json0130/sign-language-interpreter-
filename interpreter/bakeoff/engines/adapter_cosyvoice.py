"""Fun-CosyVoice3-0.5B-2512 (Alibaba/FunAudioLLM): emotion by natural-language instruction (inference_instruct2)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import HERE, REFERENCE, run_engine  # noqa: E402

REPO = HERE / "repos" / "CosyVoice"
MODEL_DIR = HERE / "models" / "cosyvoice3"
PROMPT_WAV = REFERENCE    # shared neutral reference (the repo's own prompt sounds "happy" to the SER)
_A = "You are a helpful assistant."
# emotion -> instruction. The README/utils list the tested emotion instructions as Chinese phrases
# (very happy / very sad / very angry); there is no "excited" instruction, so excited reuses "very happy".
EMOTION_MAP = {
    "neutral": f"{_A}<|endofprompt|>",
    "sad": f"{_A} 请非常伤心地说一句话。<|endofprompt|>",
    "excited": f"{_A} 请非常开心地说一句话。<|endofprompt|>",
    "angry": f"{_A} 请非常生气地说一句话。<|endofprompt|>",
}


def main():
    sys.path[:0] = [str(REPO), str(REPO / "third_party" / "Matcha-TTS")]
    import torch
    from cosyvoice.cli.cosyvoice import AutoModel
    cv = AutoModel(model_dir=str(MODEL_DIR))

    def synth(text, emotion):
        torch.manual_seed(0)
        parts = [o["tts_speech"] for o in cv.inference_instruct2(text, EMOTION_MAP[emotion], str(PROMPT_WAV), stream=False)]
        return torch.cat(parts, dim=1), cv.sample_rate

    run_engine("cosyvoice", synth, {"model": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512", "device": "cuda"})


if __name__ == "__main__":
    main()
