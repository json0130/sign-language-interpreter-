"""Preview TTS payloads for the sign_to_speech mock frames (strings only, nothing is spoken):
    python3 -m interpreter.run_tts_preview [--backend naoqi|polly|azure|all] [--out DIR]"""
import argparse
from pathlib import Path

from interpreter.context import StaticContext
from interpreter.mocks.mock_recogniser import load_frames
from interpreter.task1 import plan_speech
from interpreter.tts.base import BACKENDS, render

FRAMES = Path(__file__).parent / "mocks" / "frames"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=BACKENDS + ("all",), default="all")
    ap.add_argument("--out", help="write each payload to DIR/<frame_id>-<backend>.txt|.ssml")
    args = ap.parse_args(argv)
    backends = BACKENDS if args.backend == "all" else (args.backend,)
    out = Path(args.out) if args.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    ctx = StaticContext()
    for frame in load_frames(FRAMES, "sign_to_speech"):
        plan = plan_speech(frame, ctx)
        print(f"== {frame.frame_id} ({plan.mode}) {plan.utterance!r}")
        for b in backends:
            r = render(plan, b)
            print(f"[{b}] {r.payload}")
            for n in r.notes:
                print(f"    - {n}")
            if out:
                (out / f"{frame.frame_id}-{b}.{'txt' if b == 'naoqi' else 'ssml'}").write_text(
                    r.payload, encoding="utf-8")
        print()


if __name__ == "__main__":
    main()
