"""Render the sign_to_speech mock frames to WAV files with local Piper (run with the venv's Python):
    interpreter/.venv/bin/python -m interpreter.run_local [--frames DIR] [--out DIR] [--target-lang en-NZ|en-US]
                                                          [--voice NAME] [--expressiveness 0..2] [--play] [--flat]
                                                          [--sweep [SENTENCE]]
Files go to <out>/<lang>/<frame_id>.wav. --voice forces one Piper voice for every language.
--sweep renders ONE sentence once per emotion (<out>/<lang>/sweep/<emotion>.wav) so prosody can be tuned by ear."""
import argparse
from dataclasses import replace
from pathlib import Path

from interpreter.context import StaticContext
from interpreter.contracts import LANGS, Emotion, InterpretationFrame, Prosody
from interpreter.local_tts.config import MODELS_DIR
from interpreter.local_tts.piper_engine import PiperEngine
from interpreter.local_tts.playback import play
from interpreter.local_tts.render import render_speech
from interpreter.mocks.mock_recogniser import load_frames
from interpreter.task1 import Task1Config, plan_speech

HERE = Path(__file__).resolve().parent
SWEEP_SENTENCE = "My whānau are coming to the hui tomorrow."
SWEEP_CONFIDENCE = 0.9
# emotion -> canonical (valence, arousal) for the sweep
SWEEP_VA = {
    "neutral": (0.0, 0.0), "happy": (0.7, 0.4), "excited": (0.7, 0.8), "calm": (0.4, -0.5),
    "sad": (-0.7, -0.4), "apologetic": (-0.3, -0.1), "angry": (-0.7, 0.7), "anxious": (-0.5, 0.6),
    "surprised": (0.3, 0.8), "confused": (-0.2, 0.2),
}


def sweep_frames(sentence: str):
    """One frame per emotion with identical wording, so only prosody differs."""
    for label, (v, a) in SWEEP_VA.items():
        yield InterpretationFrame(
            frame_id=label, direction="sign_to_speech", speaker_id="person:sweep",
            addressee_id="person:listener", text=sentence, input_confidence=1.0,
            emotion=Emotion(label, v, a, SWEEP_CONFIDENCE))


def expressiveness_arg(text: str) -> float:
    x = float(text)
    if not 0 <= x <= 2:
        raise argparse.ArgumentTypeError("must be between 0 and 2")
    return x


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", default=str(HERE / "mocks" / "frames"))
    ap.add_argument("--out", default=str(HERE / "audio_out"))
    ap.add_argument("--target-lang", choices=[l for l in LANGS if l != "zh-CN"],
                    help="language to speak (default: each frame's own target_lang)")
    ap.add_argument("--voice", help="Piper voice name in interpreter/models/ (default: the language's voice)")
    ap.add_argument("--expressiveness", type=expressiveness_arg, default=Task1Config.expressiveness,
                    help="0 = flat, 1 = full emotion profile (default), 2 = exaggerated")
    ap.add_argument("--sweep", nargs="?", const=SWEEP_SENTENCE, metavar="SENTENCE",
                    help=f"render one sentence per emotion instead of the mock frames (default: {SWEEP_SENTENCE!r})")
    ap.add_argument("--play", action="store_true", help="play each file after rendering")
    ap.add_argument("--flat", action="store_true", help="also render <id>-flat.wav: same words, default prosody")
    args = ap.parse_args(argv)

    out = Path(args.out)
    engine = PiperEngine(MODELS_DIR / f"{args.voice}.onnx") if args.voice else None   # None: per language
    ctx, cfg = StaticContext(), Task1Config(expressiveness=args.expressiveness)

    if args.sweep is not None:
        frames, sub = list(sweep_frames(args.sweep)), "sweep"
        print(f"sweep: {args.sweep!r}  expressiveness={args.expressiveness:g}  (applied values, after clamping)")
    else:
        frames, sub = load_frames(args.frames, "sign_to_speech"), ""
        print(f"expressiveness={args.expressiveness:g}")

    print(f"{'frame':<11}{'emotion':<11}{'rate':>6}{'pitch':>7}{'volume':>8}{'duration':>10}  file")
    for frame in frames:
        plan = plan_speech(frame, ctx, cfg, target_lang=args.target_lang)
        lang = plan.voice.language
        dest = out / lang / sub
        dest.mkdir(parents=True, exist_ok=True)
        variants = [(frame.frame_id, plan)]
        if args.flat and args.sweep is None:
            variants.append((f"{frame.frame_id}-flat", replace(plan, prosody=Prosody())))
        for name, p in variants:
            r = render_speech(plan=p, engine=engine, out_path=dest / f"{name}.wav")
            pa = r.params
            print(f"{name:<11}{p.emotion_label:<11}{pa['rate']:>6.2f}{pa['pitch']:>7.2f}{pa['volume']:>8.2f}"
                  f"{r.duration_s:>9.2f}s  {r.path}")
            for n in r.notes:
                print(f"          - {n}")
            if args.play:
                play(r.path)


if __name__ == "__main__":
    main()
