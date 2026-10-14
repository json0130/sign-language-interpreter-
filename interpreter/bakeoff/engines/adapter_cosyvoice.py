"""Fun-CosyVoice3-0.5B-2512 (Alibaba/FunAudioLLM): emotion by natural-language instruction (inference_instruct2)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import HERE, REFERENCE, run_engine  # noqa: E402

REPO = HERE / "repos" / "CosyVoice"
MODEL_DIR = HERE / "models" / "cosyvoice3"
PROMPT_WAV = REFERENCE    # shared neutral reference (the repo's own prompt sounds "happy" to the SER)
_A = "You are a helpful assistant."
# emotion -> (instruction, speed). The repo's tested emotion instructions are Chinese phrases (very happy / sad / angry);
# there is no trained "excited". v1 reused "very happy" for excited and it sounded sad to Jay (and ran slower than
# neutral). v2 uses an untested but LLM-generalised phrasing, "speak in an excited, joyful tone", which was faster than
# neutral and heard as happy by the SER (see tune_cosyvoice.py for the other variants tried).
EMOTION_MAP = {
    "neutral": (f"{_A}<|endofprompt|>", 1.0),
    "sad": (f"{_A} 请非常伤心地说一句话。<|endofprompt|>", 1.0),
    "excited": (f"{_A} 请用激动欢快的语气说一句话。<|endofprompt|>", 1.0),
    "angry": (f"{_A} 请非常生气地说一句话。<|endofprompt|>", 1.0),
}

# Untested phrasings for the other emotions of our contract (CosyVoice has no numeric strength or mixing; mixing means
# describing it in words). Only "very happy / sad / angry" are instructions the repo tests. Used by engines/extras.py.
EXTRA_INSTRUCTIONS = {
    "happy": "请非常开心地说一句话。",
    "calm": "请用平静温和的语气说一句话。",
    "apologetic": "请用抱歉愧疚的语气说一句话。",
    "anxious": "请用紧张焦虑的语气说一句话。",
    "surprised": "请非常惊讶地说一句话。",
    "confused": "请用困惑疑惑的语气说一句话。",
    "mix_sad_gentle": "请用悲伤又温柔的语气说一句话。",     # a mixture, described in words
}


def main():
    sys.path[:0] = [str(REPO), str(REPO / "third_party" / "Matcha-TTS")]
    import torch
    from cosyvoice.cli.cosyvoice import AutoModel
    cv = AutoModel(model_dir=str(MODEL_DIR))

    def synth(text, emotion):
        torch.manual_seed(0)
        instruct, speed = EMOTION_MAP[emotion]
        parts = [o["tts_speech"] for o in cv.inference_instruct2(text, instruct, str(PROMPT_WAV), stream=False, speed=speed)]
        return torch.cat(parts, dim=1), cv.sample_rate

    run_engine("cosyvoice", synth, {"model": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512", "device": "cuda"})


if __name__ == "__main__":
    main()
