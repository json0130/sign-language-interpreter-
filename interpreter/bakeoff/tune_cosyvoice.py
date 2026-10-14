"""Try instruction/speed variants for CosyVoice's "excited" (it has no trained excited instruction; the v1 mapping reused
"very happy" and sounded sad to Jay, and ran slower than neutral). Run in venvs/cosyvoice; writes out/ct_<variant>/*.wav.
    PYTHONPATH=../.. venvs/cosyvoice/bin/python tune_cosyvoice.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HERE, OUT, REFERENCE, SENTENCES, save_wav  # noqa: E402

_A = "You are a helpful assistant."
# variant -> (instruction, speed). Only "very happy / sad / angry", loud, soft, slow and fast are tested instructions in
# the CosyVoice repo; the zh_excited* and en_excited variants are untested phrasings that the LLM front end may generalise.
VARIANTS = {
    "v1_happy": ("请非常开心地说一句话。", 1.0),
    "happy_speed115": ("请非常开心地说一句话。", 1.15),
    "happy_speed130": ("请非常开心地说一句话。", 1.30),
    "happy_fast_instr": ("请非常开心地说一句话，并用尽可能快的语速。", 1.0),
    "happy_loud_instr": ("请非常开心地、大声地说一句话。", 1.0),
    "zh_excited": ("请非常兴奋地说一句话。", 1.0),
    "zh_excited_joyful": ("请用激动欢快的语气说一句话。", 1.0),
    "en_excited": ("Please say this sentence in a very excited and happy voice.", 1.0),
}
TEXTS = {"s2": SENTENCES["s2"], "s3": SENTENCES["s3"]}


def main() -> None:
    repo = HERE / "repos" / "CosyVoice"
    sys.path[:0] = [str(repo), str(repo / "third_party" / "Matcha-TTS")]
    import torch
    from cosyvoice.cli.cosyvoice import AutoModel
    cv = AutoModel(model_dir=str(HERE / "models" / "cosyvoice3"))
    for name, (instr, speed) in VARIANTS.items():
        d = OUT / f"ct_{name}"
        d.mkdir(parents=True, exist_ok=True)
        for sid, text in TEXTS.items():
            torch.manual_seed(0)
            parts = [o["tts_speech"] for o in cv.inference_instruct2(
                text, f"{_A} {instr}<|endofprompt|>", str(REFERENCE), stream=False, speed=speed)]
            secs = save_wav(d / f"{sid}__excited.wav", torch.cat(parts, dim=1), cv.sample_rate)
            print(name, sid, round(secs, 2), flush=True)


if __name__ == "__main__":
    main()
