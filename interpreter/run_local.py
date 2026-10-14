"""Render the sign_to_speech mock frames to WAV files with local Piper (run with the venv's Python):
    interpreter/.venv/bin/python -m interpreter.run_local [--frames DIR] [--out DIR] [--target-lang en-NZ|en-US]
                                                          [--voice NAME] [--play] [--flat]
Files go to <out>/<lang>/<frame_id>.wav. --voice forces one Piper voice for every language."""
import argparse
from dataclasses import replace
from pathlib import Path

from interpreter.context import StaticContext
from interpreter.contracts import Prosody
from interpreter.contracts import LANGS
from interpreter.local_tts.config import MODELS_DIR
from interpreter.local_tts.piper_engine import PiperEngine
from interpreter.local_tts.playback import play
from interpreter.local_tts.render import render_speech
from interpreter.mocks.mock_recogniser import load_frames
from interpreter.task1 import plan_speech

HERE = Path(__file__).resolve().parent


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", default=str(HERE / "mocks" / "frames"))
    ap.add_argument("--out", default=str(HERE / "audio_out"))
    ap.add_argument("--target-lang", choices=[l for l in LANGS if l != "zh-CN"],
                    help="language to speak (default: each frame's own target_lang)")
    ap.add_argument("--voice", help="Piper voice name in interpreter/models/ (default: the language's voice)")
    ap.add_argument("--play", action="store_true", help="play each file after rendering")
    ap.add_argument("--flat", action="store_true", help="also render <id>-flat.wav: same words, default prosody")
    args = ap.parse_args(argv)

    out = Path(args.out)
    engine = PiperEngine(MODELS_DIR / f"{args.voice}.onnx") if args.voice else None   # None: per language
    ctx = StaticContext()

    print(f"{'frame':<11}{'emotion':<10}{'rate':>6}{'pitch':>7}{'volume':>8}{'duration':>10}  file")
    for frame in load_frames(args.frames, "sign_to_speech"):
        plan = plan_speech(frame, ctx, target_lang=args.target_lang)
        lang = plan.voice.language
        (out / lang).mkdir(parents=True, exist_ok=True)
        variants = [(frame.frame_id, plan)]
        if args.flat:
            variants.append((f"{frame.frame_id}-flat", replace(plan, prosody=Prosody())))
        for name, p in variants:
            r = render_speech(plan=p, engine=engine, out_path=out / lang / f"{name}.wav")
            pa = r.params
            print(f"{name:<11}{p.emotion_label:<10}{pa['rate']:>6.2f}{pa['pitch']:>7.2f}{pa['volume']:>8.2f}"
                  f"{r.duration_s:>9.2f}s  {r.path}")
            for n in r.notes:
                print(f"          - {n}")
            if args.play:
                play(r.path)


if __name__ == "__main__":
    main()
