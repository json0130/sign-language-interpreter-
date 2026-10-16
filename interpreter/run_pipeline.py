"""Whole pipeline, narrating every stage:  transcript -> emotion -> KG -> Task 1 plan -> TTS.

    interpreter/.venv/bin/python -m interpreter.run_pipeline --testset [FILE] --play     # the test sentences, spoken
    interpreter/.venv/bin/python -m interpreter.run_pipeline --text "..." [--listener ID] [--emotion E]
                                                             [--confidence C] [--concepts "ck:a ck:b"] --play
    interpreter/.venv/bin/python -m interpreter.run_pipeline [--demo] [--frames DIR]     # mock frame JSON files
  common: [--only t01,t05] [--dry-run] [--quiet] [--kg FILE] [--out DIR] [--target-lang en-NZ|en-US]
          [--expressiveness 0..2] [--play]

Each frame prints numbered stages with the decision made and why, printed from inside the real handlers (so the trace
shows what actually ran). --demo runs the te reo transcript frame (mocks/frames_kg). --dry-run skips the TTS and runs with
plain python3. Without --kg the demo graph (interpreter/kg/demo.py) is used. Speech is Piper, which conveys emotion
only through rate, pitch and volume; a better emotion TTS is still to be chosen.
"""
import argparse
from pathlib import Path

from interpreter.baseline import baseline_task2
from interpreter.concepts import CONCEPTS, find_kupu, gloss
from interpreter.contracts import LANGS
from interpreter.emotion import CANONICAL_VA, MIN_CONFIDENCE, match_rule, refine_emotion
from interpreter.contracts import EMOTIONS
from interpreter.kg.demo import demo_graph
from interpreter.kg.graph import Graph
from interpreter.kg.provider import KGContext
from interpreter.mocks.mock_recogniser import load_frames
from interpreter.mocks.stub_sign_generator import StubSignGenerator
from interpreter.pipeline import Pipeline
from interpreter.prosody_profiles import PROFILES, strength
from interpreter.run_local import expressiveness_arg
from interpreter.task1 import Task1Config, choose_words, plan_speech
from interpreter.testset import DEFAULT_TESTSET, load_testset, make_row

HERE = Path(__file__).resolve().parent
INDENT = " " * 15


class Tracer:
    """Prints numbered stages; does nothing when quiet."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def stage(self, n: int, total: int, name: str, *lines: str) -> None:
        if not self.enabled:
            return
        print(f"{f'[{n}/{total}] {name}':<15}{lines[0]}")
        for line in lines[1:]:
            print(INDENT + line)


def describe_graph(graph: Graph) -> None:
    print(f"KNOWLEDGE GRAPH: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    for node in graph.nodes.values():
        if node.kind != "person":
            continue
        facts = [f"culture {e.target_id.split(':')[1]}" for e in graph.out(node.id, "belongs_to_culture")]
        facts += [f"knows {e.target_id} ({e.weight:g})" for e in graph.out(node.id, "knows_kupu")]
        print(f"  {node.id:<20}{', '.join(facts) or 'no culture tag, no kupu edges'}")
    for edge in graph.edges.values():
        if edge.kind == "culture_prior":
            print(f"  {edge.source_id} prior for {edge.target_id}: {edge.weight:g}")
    print()


def make_task1(ctx: KGContext, cfg: Task1Config, target_lang, trace: Tracer):
    """The Task 1 handler: emotion backup, KG word choice, plan. Narrates each stage as it runs."""

    def task1(frame):
        lang = target_lang or frame.target_lang
        e = frame.emotion
        trace.stage(1, 5, "INPUT", f'"{frame.text}"',
                    f"speaker {frame.speaker_id} -> listener {frame.addressee_id}"
                    + (f" (setting: {frame.setting})" if frame.setting else ""),
                    f"recogniser emotion: {e.label} (confidence {e.confidence:.2f}); "
                    f"recogniser input confidence: {frame.input_confidence:.2f}")

        refined = refine_emotion(frame)
        if e.confidence >= MIN_CONFIDENCE:
            note = f"keep the recogniser's {e.label}: confidence {e.confidence:.2f} >= {MIN_CONFIDENCE:.2f}"
        elif refined is not frame:
            _, matched = match_rule(frame.text)
            v, a = CANONICAL_VA[refined.emotion.label]
            note = (f'recogniser confidence {e.confidence:.2f} < {MIN_CONFIDENCE:.2f}: text rule matched "{matched}"'
                    f" -> {refined.emotion.label} (confidence {refined.emotion.confidence:.2f}, "
                    f"valence {v:+.1f}, arousal {a:+.1f})")
        else:
            note = (f"recogniser confidence {e.confidence:.2f} < {MIN_CONFIDENCE:.2f} but no text rule matched: "
                    f"keep {e.label}")
        trace.stage(2, 5, "EMOTION", note)

        if frame.input_confidence < cfg.clarify_threshold:
            trace.stage(3, 5, "KG", "skipped: this frame becomes a clarification request")
        else:
            slugs = {c.slug for c in frame.concepts if c.te_reo} | {s for s, _ in find_kupu(frame.text)}
            lines = []
            for slug in sorted(slugs):
                knows, why = ctx.explain(frame.addressee_id, slug)
                entry = CONCEPTS.get(slug)
                spoken = entry.te_reo if knows and entry else (gloss(slug, lang) or slug)
                lines.append(f'{slug}: listener {"KNOWS" if knows else "does NOT know"} it ({why}) -> say "{spoken}"')
            if not lines:
                lines = ["no kupu Māori or tagged concepts in this transcript: nothing to look up"]
            changes = [f'"{old}" -> "{new}"' for old, new, _ in choose_words(refined, ctx, lang)]
            lines.append("word changes: " + (", ".join(changes) if changes else "none"))
            trace.stage(3, 5, "KG", *lines)

        plan = plan_speech(refined, ctx, cfg, target_lang=target_lang)
        p = plan.prosody
        if plan.mode == "clarify":
            trace.stage(4, 5, "PLAN", f"clarify: input confidence {frame.input_confidence:.2f} < "
                        f"{cfg.clarify_threshold:.2f}, so Pepper asks for a repeat in its robot voice",
                        f'says: "{plan.utterance}"')
        else:
            em = refined.emotion
            s = strength(cfg.expressiveness, em.valence, em.arousal, em.confidence)
            pr, pp, pv = PROFILES[em.label]
            extras = []
            if p.emphasis:
                extras.append("stress " + ", ".join(p.emphasis))
            if p.pauses:
                extras.append("pauses " + ", ".join(f"{x.ms} ms after word {x.after_word_index}" for x in p.pauses))
            if plan.lexicon:
                extras.append("pronunciation: " + ", ".join(f'{x.word} -> "{x.respelling}"' for x in plan.lexicon))
            trace.stage(4, 5, "PLAN", f'says: "{plan.utterance}"',
                        f"delivery: {em.label} profile (rate {pr:g}, pitch {pp:g}, volume {pv:g}) x strength {s:.2f} "
                        f"[expressiveness {cfg.expressiveness:g}, emotion intensity and confidence]",
                        f"  -> rate {p.rate:.2f}, pitch {p.pitch:.2f}, volume {p.volume:.2f}",
                        *(["extras: " + "; ".join(extras)] if extras else []),
                        f"eye colour {plan.eye.rgb} {plan.eye.pattern} (the local TTS ignores this)")
        return plan

    return task1


def make_task2(trace: Tracer):
    def task2(frame):
        trace.stage(1, 2, "INPUT", f'"{frame.text}"', f"speaker {frame.speaker_id} -> {frame.addressee_id}, "
                    f"emotion {frame.emotion.label} ({frame.emotion.confidence:.2f})")
        plan = baseline_task2(frame)
        trace.stage(2, 2, "TASK 2", "baseline only (the speech-to-sign planner is not built yet): whole transcript "
                    "passed through as one unit")
        return plan

    return task2


class Recorder:
    """Silent stand-in speech sink for --dry-run."""

    def __init__(self) -> None:
        self.received = []

    def speak(self, plan) -> None:
        self.received.append(plan)


def build_pipeline(graph: Graph, speech_sink, sign_sink, cfg: Task1Config, target_lang=None,
                   trace: Tracer = Tracer(False)) -> Pipeline:
    ctx = KGContext(graph)
    return Pipeline(make_task1(ctx, cfg, target_lang, trace), make_task2(trace), speech_sink, sign_sink)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", default=str(HERE / "mocks" / "frames"))
    ap.add_argument("--demo", action="store_true", help="run the te reo transcript frame (mocks/frames_kg)")
    ap.add_argument("--testset", nargs="?", const=str(DEFAULT_TESTSET), metavar="CSV",
                    help=f"run the sentences in a test-set CSV (default {DEFAULT_TESTSET.name})")
    ap.add_argument("--text", help="run one sentence typed on the command line")
    ap.add_argument("--listener", default="person:visitor_01", help="with --text: the addressee id in the KG")
    ap.add_argument("--emotion", default="neutral", choices=EMOTIONS, help="with --text: the recogniser's emotion")
    ap.add_argument("--confidence", type=float, default=0.8, help="with --text: the recogniser's emotion confidence")
    ap.add_argument("--concepts", default="", help='with --text: tagged concepts, e.g. "ck:whanau ck:hui"')
    ap.add_argument("--only", help="comma-separated frame ids to run, e.g. f-001,f-011")
    ap.add_argument("--kg", help="JSON graph saved by Graph.save (default: the demo graph)")
    ap.add_argument("--out", default=str(HERE / "audio_out" / "pipeline"))
    ap.add_argument("--target-lang", choices=[l for l in LANGS if l != "zh-CN"])
    ap.add_argument("--expressiveness", type=expressiveness_arg, default=Task1Config.expressiveness)
    ap.add_argument("--play", action="store_true", help="play each WAV after rendering")
    ap.add_argument("--dry-run", action="store_true", help="print the stages only; no TTS")
    ap.add_argument("--quiet", action="store_true", help="one line in, one line out per frame")
    args = ap.parse_args(argv)

    verbose = not args.quiet
    trace = Tracer(verbose)
    graph = Graph.load(args.kg) if args.kg else demo_graph()
    if args.dry_run:
        sink = Recorder()
    else:
        from interpreter.local_tts.sink import LocalSpeechSink   # needs the venv (piper), so imported lazily
        sink = LocalSpeechSink(args.out)
        from interpreter.local_tts.playback import play
    cfg = Task1Config(expressiveness=args.expressiveness)
    sign_sink = StubSignGenerator()
    pipe = build_pipeline(graph, sink, sign_sink, cfg, args.target_lang, trace)

    notes = {}
    if args.text:
        rows = [make_row("cli", args.text, args.emotion, args.confidence, args.listener, args.concepts)]
    elif args.testset:
        rows = load_testset(args.testset)
    else:
        rows = None
    if rows is not None:
        frames = [r.frame for r in rows]
        notes = {r.frame.frame_id: r.notes for r in rows if r.notes}
    else:
        frames = load_frames(str(HERE / "mocks" / "frames_kg") if args.demo else args.frames)
    if args.only:
        wanted = set(args.only.split(","))
        frames = [f for f in frames if f.frame_id in wanted]
        if not frames:
            raise SystemExit(f"no frame matches --only {args.only}")

    if verbose:
        describe_graph(graph)
    spoken = signed = 0
    summary = []
    for frame in frames:
        to_speech = frame.direction in ("sign_to_speech", "text_to_speech")
        print(f"\n===== {frame.frame_id}  {frame.direction}  ->  {'Task 1 (speech)' if to_speech else 'Task 2 (sign)'} =====")
        if frame.frame_id in notes and verbose:
            print(f"listen for: {notes[frame.frame_id]}")
        plan = pipe.process(frame)
        if to_speech:
            spoken += 1
            if args.dry_run:
                trace.stage(5, 5, "SPEECH", "dry run: no audio rendered")
            else:
                r = sink.results[-1]
                trace.stage(5, 5, "SPEECH", f"voice {plan.voice.language}, {r.duration_s:.2f} s of audio, "
                            f"applied rate {r.params['rate']:.2f} pitch {r.params['pitch']:.2f} "
                            f"volume {r.params['volume']:.2f}", *(f"note: {n}" for n in r.notes), f"file: {r.path}")
                if args.play:
                    trace.stage(5, 5, "SPEECH", "playing...")
                    play(r.path)
            weak = frame.emotion.confidence < MIN_CONFIDENCE
            source = "text rule" if refine_emotion(frame) is not frame else ("recogniser (weak)" if weak else "recogniser")
            summary.append((frame.frame_id, frame.addressee_id, f"{plan.emotion_label} ({source})", plan.utterance))
            if not verbose:
                print(f'  in : "{frame.text}"\n  out: "{plan.utterance}"')
        else:
            signed += 1
    if summary:
        print("\nSUMMARY")
        print(f"  {'id':<7}{'listener':<19}{'emotion (source)':<30}says")
        for row in summary:
            print(f"  {row[0]:<7}{row[1]:<19}{row[2]:<30}{row[3]}")
    print(f"\ndone: {spoken} speech plan(s), {signed} sign plan(s)" + ("" if args.dry_run else f"; WAVs in {args.out}"))


if __name__ == "__main__":
    main()
