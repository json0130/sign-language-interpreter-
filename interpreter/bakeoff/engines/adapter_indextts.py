"""IndexTTS-2.5 (bilibili): emotion by an 8-value emotion vector, separate from the speaker reference."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import HERE, REFERENCE, run_engine  # noqa: E402

REPO = HERE / "repos" / "index-tts"
MODEL_DIR = HERE / "models" / "indextts-2.5"
SPEAKER_WAV = REFERENCE    # shared neutral reference (the repo's voice_01.wav sounds "angry" to the SER)
# emotion -> emo_vector [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]: any weights, so emotions
# can be MIXED. IndexTTS then down-weights some (happy .94, angry .88, disgusted .94, melancholic .94, surprised .69,
# calm .56) and rescales so the total is at most 0.8. Starting values to tune by ear.
# IndexTTS's own neutral class is "calm" (its text-emotion model maps 自然 to calm). A first run used emo_vector=None for
# neutral, which takes the emotion from the speaker reference clip; the SER heard that clip as angry, so neutral is an
# explicit calm vector. The first four are the bake-off emotions; the other six complete our contract's ten (untuned).
EMOTION_MAP = {
    "neutral": [0, 0, 0, 0, 0, 0, 0, 0.8],
    "sad": [0, 0, 0.8, 0, 0, 0.2, 0, 0],
    "excited": [0.7, 0, 0, 0, 0, 0, 0.3, 0],     # no excited slot: happy + surprised
    "angry": [0, 0.8, 0, 0, 0, 0, 0, 0],
    "happy": [0.8, 0, 0, 0, 0, 0, 0, 0],
    "calm": [0, 0, 0, 0, 0, 0, 0, 0.9],
    "apologetic": [0, 0, 0.4, 0, 0, 0.3, 0, 0.4],  # a mix: sad + gentle sadness + calm
    "anxious": [0, 0, 0, 0.6, 0, 0, 0.2, 0],       # afraid + a little surprised
    "surprised": [0, 0, 0, 0, 0, 0, 0.8, 0],
    "confused": [0, 0, 0, 0.2, 0, 0.1, 0.4, 0],    # no confused slot: approximate
}
# emotion -> emo_alpha (multiplies the whole vector, 0..1). Jay's picks after listening to 0.3/0.5/0.7/1.0 (1.0 was
# "a bit too exaggerated"): neutral 0.5, sad 0.7, excited 0.5, angry 0.7. The other six are untuned guesses.
ALPHA_BY_EMOTION = {"neutral": 0.5, "sad": 0.7, "excited": 0.5, "angry": 0.7,
                    "happy": 0.5, "calm": 0.5, "apologetic": 0.5, "anxious": 0.7, "surprised": 0.5, "confused": 0.5}


def main():
    # argv[1]: nothing = alpha 1.0 (round 1); a number in (0, 1] = one alpha for every emotion
    # (out/indextts_a<percent>/); "mixed" = ALPHA_BY_EMOTION (out/indextts_mixed/).
    arg = sys.argv[1] if len(sys.argv) > 1 else "1.0"
    alpha_for = ALPHA_BY_EMOTION.get if arg == "mixed" else (lambda e: float(arg))
    name = "indextts_mixed" if arg == "mixed" else ("indextts" if float(arg) == 1.0 else f"indextts_a{round(float(arg) * 100)}")
    sys.path.insert(0, str(REPO))
    import numpy as np
    import torch
    from indextts.infer_v2_5 import IndexTTS2
    tts = IndexTTS2(cfg_path=str(MODEL_DIR / "config.yaml"), model_dir=str(MODEL_DIR), use_bf16=True)

    def synth(text, emotion):
        torch.manual_seed(0)
        vec = EMOTION_MAP[emotion]
        sr, wav = tts.infer(spk_audio_prompt=str(SPEAKER_WAV), text=text, output_path=None, lang="EN",
                            emo_vector=vec, emo_alpha=alpha_for(emotion), use_random=False)
        return np.asarray(wav, dtype="float32").reshape(-1) / 32768.0, sr

    run_engine(name, synth, {"model": "IndexTeam/IndexTTS-2.5", "device": "cuda", "emo_alpha": arg})


if __name__ == "__main__":
    main()
