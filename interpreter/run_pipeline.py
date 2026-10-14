"""Whole pipeline on the mock frames:  transcript -> emotion backup -> KG context -> Task 1 plan -> local TTS.

    interpreter/.venv/bin/python -m interpreter.run_pipeline [--kg FILE] [--out DIR] [--target-lang en-NZ|en-US]
                                                             [--expressiveness 0..2] [--play] [--dry-run]

Without --kg the demo graph (interpreter/kg/demo.py) is used. --dry-run prints the plans and skips the TTS, so it runs
with plain python3. Speech uses Piper, which conveys emotion only through rate/pitch/volume; a better emotion TTS is
still to be chosen.
"""
import argparse
from pathlib import Path

from interpreter.baseline import baseline_task2
from interpreter.contracts import LANGS
from interpreter.emotion import refine_emotion
from interpreter.kg.demo import demo_graph
from interpreter.kg.graph import Graph
from interpreter.kg.provider import KGContext
from interpreter.mocks.mock_pepper import MockPepper
from interpreter.mocks.mock_recogniser import load_frames
from interpreter.mocks.stub_sign_generator import StubSignGenerator
from interpreter.pipeline import Pipeline
from interpreter.run_local import expressiveness_arg
from interpreter.task1 import Task1Config, plan_speech

HERE = Path(__file__).resolve().parent


def build_pipeline(graph: Graph, speech_sink, sign_sink, cfg: Task1Config, target_lang=None) -> Pipeline:
    ctx = KGContext(graph)

    def task1(frame):
        return plan_speech(refine_emotion(frame), ctx, cfg, target_lang=target_lang)

    return Pipeline(task1, baseline_task2, speech_sink, sign_sink)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", default=str(HERE / "mocks" / "frames"))
    ap.add_argument("--kg", help="JSON graph saved by Graph.save (default: the demo graph)")
    ap.add_argument("--out", default=str(HERE / "audio_out" / "pipeline"))
    ap.add_argument("--target-lang", choices=[l for l in LANGS if l != "zh-CN"])
    ap.add_argument("--expressiveness", type=expressiveness_arg, default=Task1Config.expressiveness)
    ap.add_argument("--play", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print plans only; no TTS")
    args = ap.parse_args(argv)

    graph = Graph.load(args.kg) if args.kg else demo_graph()
    if args.dry_run:
        sink = MockPepper()
    else:
        from interpreter.local_tts.sink import LocalSpeechSink   # needs the venv (piper), so imported lazily
        sink = LocalSpeechSink(args.out, play_audio=args.play)
    pipe = build_pipeline(graph, sink, StubSignGenerator(), Task1Config(expressiveness=args.expressiveness),
                          args.target_lang)

    for frame in load_frames(args.frames):
        refined = refine_emotion(frame)
        print(f"\n{frame.frame_id} {frame.speaker_id} -> {frame.addressee_id}")
        print(f"  in : {frame.text!r}  emotion {frame.emotion.label} ({frame.emotion.confidence:.2f})"
              + (f" -> {refined.emotion.label} (text rules)" if refined is not frame else ""))
        plan = pipe.process(frame)
        if frame.direction == "sign_to_speech":
            print(f"  out: {plan.utterance!r}  rate {plan.prosody.rate:.2f} pitch {plan.prosody.pitch:.2f} "
                  f"vol {plan.prosody.volume:.2f}")
            if not args.dry_run:
                print(f"       {sink.results[-1].path}")


if __name__ == "__main__":
    main()
