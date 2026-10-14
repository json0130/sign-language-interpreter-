"""IndexTTS-2.5 (bilibili): emotion by an 8-value emotion vector, separate from the speaker reference."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import HERE, REFERENCE, run_engine  # noqa: E402

REPO = HERE / "repos" / "index-tts"
MODEL_DIR = HERE / "models" / "indextts-2.5"
SPEAKER_WAV = REFERENCE    # shared neutral reference (the repo's voice_01.wav sounds "angry" to the SER)
# emotion -> emo_vector [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]. Starting values to
# tune by ear. IndexTTS's own neutral class is "calm" (its text-emotion model maps 自然 to calm). A first run used
# emo_vector=None for neutral, which takes the emotion from the speaker reference clip; that clip sounded angry to the
# SER, so neutral was changed to an explicit calm vector. "excited" has no slot, so it is happy + surprised.
EMOTION_MAP = {
    "neutral": [0, 0, 0, 0, 0, 0, 0, 0.8],
    "sad": [0, 0, 0.8, 0, 0, 0.2, 0, 0],
    "excited": [0.7, 0, 0, 0, 0, 0, 0.3, 0],
    "angry": [0, 0.8, 0, 0, 0, 0, 0, 0],
}


def main():
    sys.path.insert(0, str(REPO))
    import numpy as np
    import torch
    from indextts.infer_v2_5 import IndexTTS2
    tts = IndexTTS2(cfg_path=str(MODEL_DIR / "config.yaml"), model_dir=str(MODEL_DIR), use_bf16=True)

    def synth(text, emotion):
        torch.manual_seed(0)
        vec = EMOTION_MAP[emotion]
        sr, wav = tts.infer(spk_audio_prompt=str(SPEAKER_WAV), text=text, output_path=None, lang="EN",
                            emo_vector=vec, use_random=False)
        return np.asarray(wav, dtype="float32").reshape(-1) / 32768.0, sr

    run_engine("indextts", synth, {"model": "IndexTeam/IndexTTS-2.5", "device": "cuda"})


if __name__ == "__main__":
    main()
