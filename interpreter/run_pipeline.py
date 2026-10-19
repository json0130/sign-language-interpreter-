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
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

from interpreter.baseline import baseline_task2
from interpreter.concepts import CONCEPTS, find_kupu, gloss
from interpreter.contracts import EMOTIONS, LANGS, SpeechPlan
from interpreter.emotion import CANONICAL_VA, MIN_CONFIDENCE, match_rule, refine_emotion
from interpreter.emotion_flow import SpeakerState, describe
from interpreter.kg.demo import demo_graph
from interpreter.kg.graph import Graph
from interpreter.kg.provider import KGContext
from interpreter.mocks.mock_recogniser import load_frames
from interpreter.mocks.stub_sign_generator import StubSignGenerator
from interpreter.pipeline import Pipeline
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


def make_task1(ctx: KGContext, cfg: Task1Config, target_lang, trace: Tracer, state: Optional[SpeakerState] = None):
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

        previous = state.get(frame.speaker_id) if state else None
        plan = plan_speech(refined, ctx, cfg, target_lang=target_lang, previous=previous)
        if state and plan.segments:
            state.update(frame.speaker_id, plan.segments[-1].valence, plan.segments[-1].arousal)
        p = plan.prosody
        if plan.mode == "clarify":
            trace.stage(4, 5, "PLAN", f"clarify: input confidence {frame.input_confidence:.2f} < "
                        f"{cfg.clarify_threshold:.2f}, so Pepper asks for a repeat in its robot voice",
                        f'says: "{plan.utterance}"')
        else:
            em = refined.emotion
            spans = ", ".join(f"word {x.start_word}: {x.emotion.label}" for x in refined.emotion_flow)
            extras = []
            if p.emphasis:
                extras.append("stress " + ", ".join(p.emphasis))
            trace.stage(4, 5, "PLAN", f'says: "{plan.utterance}"',
                        f"sentence emotion {em.label} (confidence {em.confidence:.2f})"
                        + (f"; recogniser emotion changes at {spans}" if spans else "")
                        + (f"; carried from this speaker's last sentence: valence {previous[0]:+.2f}, "
                           f"arousal {previous[1]:+.2f}" if previous else ""),
                        f"emotion flow ({len(plan.segments)} clause{'s' if len(plan.segments) != 1 else ''}):",
                        *("  " + line for line in describe(plan)),
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


def emotion_source(frame) -> str:
    if refine_emotion(frame) is not frame:
        return "text rule"
    return "recogniser (weak)" if frame.emotion.confidence < MIN_CONFIDENCE else "recogniser"


# ---------------------------------------------------------------- one sentence at a time, and scoring
NAV_PROMPT = "Enter/n = next, p = previous, r = replay, q = quit, or type an id: "


def step_through(frames, run_one, play_wav, scorer=None) -> None:
    """Run, play (and score) one frame at a time, letting the listener move around. Ends on q, Ctrl-D, Ctrl-C, or
    Enter after the last frame. Revisiting a frame runs it again (and re-scores it if scoring)."""
    ids = [f.frame_id for f in frames]
    i = 0
    try:
        while True:
            plan, wav = run_one(frames[i])
            play_wav(wav)
            if scorer and isinstance(plan, SpeechPlan):
                scorer.ask(frames[i], plan, lambda: play_wav(wav))
            while True:
                cmd = input(f"[{i + 1}/{len(frames)}] {NAV_PROMPT}").strip()
                if cmd.lower() in ("", "n"):
                    if i + 1 == len(frames):
                        return
                    i += 1
                    break
                if cmd.lower() == "p":
                    if i == 0:
                        print("this is the first one")
                        continue
                    i -= 1
                    break
                if cmd.lower() == "r":
                    play_wav(wav)
                    continue
                if cmd.lower() == "q":
                    return
                if cmd in ids:
                    i = ids.index(cmd)
                    break
                print(f"not understood: {cmd!r}")
    except (EOFError, KeyboardInterrupt):
        print("\nstopped")


# The marking scheme. "single" checks apply to one-clause sentences, "multi" to sentences with an emotion flow.
CHECKS = [
    ("words", "Words: what you heard matches the says: line exactly", "all"),
    ("maori", "Māori words: none for an English listener; kept only where the KG says the listener knows them", "all"),
    ("emotion", "Emotion: with your eyes closed you would name the intended emotion", "single"),
    ("flow", "Flow: the change at each clause boundary is audible and goes the right way", "multi"),
    ("natural", "Natural: sounds like a real person (not robotic, chipmunky, dragging or rushed)", "all"),
    ("joins", "Joins: no clicks, odd gaps or volume jumps between clauses", "multi"),
    ("not_overdone", "Not overdone: the emotion is not cartoonish or exaggerated", "all"),
]
DEFAULT_SCORES = HERE / "listening_results" / "scores.csv"
ANSWERS = {"y": "yes", "n": "no", "s": "skipped"}


class Scorer:
    """Asks the marking-scheme questions after a sentence plays and keeps the answers in a CSV (rewritten after
    every sentence, so nothing is lost if you stop early). With the default path a new timestamped file is used."""

    FIELDS = ["timestamp", "id", "voice", "listener", "says", *[k for k, _, _ in CHECKS], "verdict", "comment"]

    def __init__(self, path: Path, voice: str) -> None:
        if path == DEFAULT_SCORES:
            path = path.with_name(f"scores-{voice.replace(' ', '-')}-{datetime.now():%Y%m%d-%H%M%S}.csv")
        self.path, self.voice = path, voice
        self.rows = {}

    def ask(self, frame, plan, replay) -> str:
        multi = len(plan.segments) > 1
        print("SCORE    y = yes, n = no, s = skip, r = replay the audio")
        answers = {}
        for key, question, applies in CHECKS:
            if (applies == "single" and multi) or (applies == "multi" and not multi):
                answers[key] = "n/a"
                continue
            while True:
                a = input(f"  {question}? ").strip().lower()
                if a == "r":
                    replay()
                elif a in ANSWERS:
                    answers[key] = ANSWERS[a]
                    break
                else:
                    print("  y, n, s or r")
        comment = input("  comment (Enter for none): ").strip()
        given = [v for v in answers.values() if v != "n/a"]
        verdict = "no" if "no" in given else ("incomplete" if "skipped" in given else "yes")
        self.rows[frame.frame_id] = {"timestamp": datetime.now().isoformat(timespec="seconds"), "id": frame.frame_id,
                                     "voice": self.voice, "listener": frame.addressee_id, "says": plan.utterance,
                                     **answers, "verdict": verdict, "comment": comment}
        self.save()
        print(f"  -> {verdict} (saved to {self.path})")
        return verdict

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=self.FIELDS)
            w.writeheader()
            w.writerows(self.rows.values())

    def report(self) -> None:
        if not self.rows:
            print("\nSCORES: nothing scored")
            return
        rows = list(self.rows.values())
        passed = sum(r["verdict"] == "yes" for r in rows)
        print(f"\nSCORES ({self.voice}): {passed}/{len(rows)} sentences passed every check")
        for key, question, _ in CHECKS:
            answered = [r[key] for r in rows if r[key] in ("yes", "no")]
            if answered:
                print(f"  {question.split(':')[0]:<14}{answered.count('yes')}/{len(answered)} yes")
        failed = [r["id"] for r in rows if r["verdict"] == "no"]
        if failed:
            print(f"  failed: {', '.join(failed)}")
        print(f"  saved to {self.path}")


def build_pipeline(graph: Graph, speech_sink, sign_sink, cfg: Task1Config, target_lang=None,
                   trace: Tracer = Tracer(False), state: Optional[SpeakerState] = None) -> Pipeline:
    """state: a SpeakerState to carry each speaker's emotion from one sentence to the next (None = off)."""
    ctx = KGContext(graph)
    return Pipeline(make_task1(ctx, cfg, target_lang, trace, state), make_task2(trace), speech_sink, sign_sink)


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
    ap.add_argument("--tts", choices=("kokoro", "piper"), help="local voice engine (default: config.TTS_ENGINE, kokoro)")
    ap.add_argument("--voice", help="a Kokoro voice for every sentence, e.g. af_heart, af_bella, bf_emma, bf_isabella")
    ap.add_argument("--carry", action="store_true",
                    help="carry each speaker's emotion from one sentence to the next (off: every sentence starts fresh)")
    ap.add_argument("--step", action="store_true",
                    help="one sentence at a time: plays it, then Enter/n = next, p = previous, r = replay, q = quit, "
                         "or type an id to jump")
    ap.add_argument("--score", nargs="?", const=str(DEFAULT_SCORES), metavar="CSV",
                    help="like --step, and asks the marking-scheme questions after each sentence; answers are saved "
                         "to a CSV (default: a new file in interpreter/listening_results/)")
    ap.add_argument("--dry-run", action="store_true", help="print the stages only; no TTS")
    ap.add_argument("--quiet", action="store_true", help="one line in, one line out per frame")
    args = ap.parse_args(argv)

    verbose = not args.quiet
    if args.dry_run:
        def play(_):
            pass
    else:
        from interpreter.local_tts.playback import play
    trace = Tracer(verbose)
    graph = Graph.load(args.kg) if args.kg else demo_graph()
    if args.dry_run:
        sink = Recorder()
    else:
        from interpreter.local_tts.sink import LocalSpeechSink   # needs the venv (piper), so imported lazily
        engine = None
        if args.voice:
            from interpreter.local_tts.kokoro_engine import KokoroEngine
            engine = KokoroEngine(args.voice)
        sink = LocalSpeechSink(args.out, engine=engine, tts=args.tts)
    cfg = Task1Config(expressiveness=args.expressiveness)
    sign_sink = StubSignGenerator()
    pipe = build_pipeline(graph, sink, sign_sink, cfg, args.target_lang, trace, SpeakerState() if args.carry else None)

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
    voice = "dry-run" if args.dry_run else sink.voice_name(args.target_lang or frames[0].target_lang)
    results = {}                                  # frame id -> (frame, plan, wav path or None), in visiting order

    def run_one(frame):
        to_speech = frame.direction in ("sign_to_speech", "text_to_speech")
        print(f"\n===== {frame.frame_id}  {frame.direction}  ->  {'Task 1 (speech)' if to_speech else 'Task 2 (sign)'} =====")
        if frame.frame_id in notes and verbose:
            print(f"listen for: {notes[frame.frame_id]}")
        plan, wav = pipe.process(frame), None
        if to_speech:
            if args.dry_run:
                trace.stage(5, 5, "SPEECH", "dry run: no audio rendered")
            else:
                r = sink.results[-1]
                wav = r.path
                parts = r.params.get("segments") or [{**r.params, "duration_s": r.duration_s}]
                trace.stage(5, 5, "SPEECH", f"{sink.voice_name(plan.voice.language)} ({plan.voice.language}), "
                            f"{r.duration_s:.2f} s of audio",
                            *(f"clause {k}: rate {x['rate']:.2f} pitch {x['pitch']:.2f} volume {x['volume']:.2f}, "
                              f"{x['duration_s']:.2f} s" for k, x in enumerate(parts, start=1)),
                            *(f"note: {n}" for n in r.notes), f"file: {r.path}")
            if not verbose:
                print(f'  in : "{frame.text}"\n  out: "{plan.utterance}"')
        results[frame.frame_id] = (frame, plan, wav)
        return plan, wav

    def play_wav(wav):
        if wav:
            trace.stage(5, 5, "SPEECH", "playing...")
            play(wav)
        else:
            print("(no audio: dry run or a sign frame)")

    scorer = Scorer(Path(args.score), voice) if args.score else None
    if args.step or scorer:
        step_through(frames, run_one, play_wav, scorer)
    else:
        for frame in frames:
            _, wav = run_one(frame)
            if args.play and wav:
                play_wav(wav)

    summary = [(f.frame_id, f.addressee_id, f"{p.emotion_label} ({emotion_source(f)})", p.utterance)
               for f, p, _ in results.values() if isinstance(p, SpeechPlan)]
    if summary:
        print("\nSUMMARY")
        print(f"  {'id':<7}{'listener':<19}{'emotion (source)':<30}says")
        for row in summary:
            print(f"  {row[0]:<7}{row[1]:<19}{row[2]:<30}{row[3]}")
    if scorer:
        scorer.report()
    spoken = len(summary)
    print(f"\ndone: {spoken} speech plan(s), {len(results) - spoken} sign plan(s)"
          + ("" if args.dry_run else f"; WAVs in {args.out}"))

if __name__ == "__main__":
    main()
