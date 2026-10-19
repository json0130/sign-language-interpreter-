"""Emotion flow: one emotion per clause instead of one per sentence. Task 1 uses it to give each clause its own delivery;
it also runs on its own for A/B listening (the same sentence with one emotion, then with the flow):

    python3 -m interpreter.emotion_flow --testset [CSV] [--only t12,t13]          # print the flow per sentence
    python3 -m interpreter.emotion_flow --text "{happy}Kia ora, {sad}but my nan could not come." [--emotion E]
    interpreter/.venv/bin/python -m interpreter.emotion_flow --testset --only t12 --play   # play single, then flow
      [--render] [--carry] [--tts kokoro|indextts|piper] [--expressiveness 0..2] [--target-lang en-NZ|en-US] [--out DIR]

How a clause gets its emotion, in order:
  1. the recogniser's emotion span covering it (frame.emotion_flow; in the test CSV, {emotion} markers), used as given;
  2. otherwise a text rule (emotion.py) matching the clause, blended with the sentence emotion by their confidences;
  3. otherwise the sentence emotion.
Rule- and sentence-driven clauses keep 30% of the previous clause (inertia), so the voice does not jump. With carry-over,
the first clause also keeps 30% of the speaker's state at the end of their previous sentence.
Clauses break at sentence punctuation (. ! ? ;), before ", but" / ", so" / ", although" / ", and then", and where a
recogniser span starts. Emotion is held as valence/arousal; delivery (rate, pitch, volume) is a distance-weighted blend of
the emotion profiles, so in-between emotions get in-between delivery. Starting values to tune by ear.
"""
import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from interpreter.contracts import EMOTIONS, LANGS, Emotion, EmotionSpan
from interpreter.emotion import CANONICAL_VA, RULE_CONFIDENCE, match_rule
from interpreter.prosody_profiles import CLAMPS, PROFILES, strength

HERE = Path(__file__).resolve().parent
MARKER = re.compile(r"\{([a-z]+)\}")
JOINERS = ("but", "so", "although", "though")
MARKER_CONFIDENCE = 0.9  # how sure an authored {marker} is (it stands in for a confident recogniser reading)
INERTIA = 0.3            # share of the previous clause kept by a rule- or sentence-driven clause
STATE_CARRY = 0.3        # share of the speaker's previous state kept by the first clause (carry-over)


@dataclass(frozen=True)
class Clause:
    start: int           # index of its first word in text.split()
    text: str


@dataclass(frozen=True)
class Segment:
    text: str
    valence: float
    arousal: float
    confidence: float
    source: str          # "recogniser", "rule" or "utterance" (contracts.SEGMENT_SOURCES)

    @property
    def label(self) -> str:
        return nearest_label(self.valence, self.arousal)


def nearest_label(valence: float, arousal: float) -> str:
    return min(CANONICAL_VA, key=lambda l: (CANONICAL_VA[l][0] - valence) ** 2 + (CANONICAL_VA[l][1] - arousal) ** 2)


def markers_to_spans(marked_text: str, confidence: float = MARKER_CONFIDENCE) -> Tuple[str, List[EmotionSpan]]:
    """'{happy}Kia ora, {sad}but ...' -> ('Kia ora, but ...', [span at word 0 happy, span at word 2 sad])."""
    spans: Dict[int, EmotionSpan] = {}
    for m in MARKER.finditer(marked_text):
        label = m.group(1)
        if label not in EMOTIONS:
            raise ValueError(f"unknown emotion marker {{{label}}}; use one of {', '.join(EMOTIONS)}")
        start = len(MARKER.sub("", marked_text[:m.start()]).split())
        spans[start] = EmotionSpan(start, Emotion(label, *CANONICAL_VA[label], confidence))
    plain = " ".join(MARKER.sub(" ", marked_text).split())
    return plain, [spans[k] for k in sorted(spans)]


def clauses(text: str, breaks: Iterable[int] = ()) -> List[Clause]:
    words, forced, out, start = text.split(), set(breaks), [], 0
    for i in range(1, len(words)):
        prev, cur = words[i - 1], words[i].lower().strip(".,!?;:")
        joiner = cur in JOINERS or (cur == "and" and i + 1 < len(words) and words[i + 1].lower().startswith("then"))
        if i in forced or prev[-1] in ".!?;" or (prev.endswith(",") and joiner):
            out.append(Clause(start, " ".join(words[start:i])))
            start = i
    if words:
        out.append(Clause(start, " ".join(words[start:])))
    return out


def _clamp(x: float) -> float:
    return max(-1.0, min(1.0, x))


def flow(text: str, utterance: Emotion, spans: Iterable[EmotionSpan] = (),
         previous: Optional[Tuple[float, float]] = None) -> List[Segment]:
    """Per-clause emotions for one sentence. `previous` is the speaker's last (valence, arousal), for carry-over."""
    spans = sorted(spans, key=lambda s: s.start_word)
    segments: List[Segment] = []
    for clause in clauses(text, (s.start_word for s in spans)):
        span = next((s for s in reversed(spans) if s.start_word <= clause.start), None)
        if span is not None:
            e = span.emotion
            v, a, conf, source = e.valence, e.arousal, e.confidence, "recogniser"
        else:
            rule = match_rule(clause.text)
            if rule:
                rv, ra = CANONICAL_VA[rule[0]]
                cu, cr = utterance.confidence, RULE_CONFIDENCE
                if cu + cr:
                    v, a = (cu * utterance.valence + cr * rv) / (cu + cr), (cu * utterance.arousal + cr * ra) / (cu + cr)
                else:
                    v, a = rv, ra
                conf, source = max(cu, cr), "rule"
            else:
                v, a, conf, source = utterance.valence, utterance.arousal, utterance.confidence, "utterance"
            anchor = (segments[-1].valence, segments[-1].arousal) if segments else previous
            if anchor is not None:
                keep = INERTIA if segments else STATE_CARRY
                v, a = (1 - keep) * v + keep * anchor[0], (1 - keep) * a + keep * anchor[1]
        segments.append(Segment(clause.text, round(_clamp(v), 3), round(_clamp(a), 3), conf, source))
    return segments


class SpeakerState:
    """Each speaker's (valence, arousal) at the end of their last sentence."""

    def __init__(self) -> None:
        self.last: Dict[str, Tuple[float, float]] = {}

    def get(self, speaker_id: str) -> Optional[Tuple[float, float]]:
        return self.last.get(speaker_id)

    def update(self, speaker_id: str, valence: float, arousal: float) -> None:
        self.last[speaker_id] = (valence, arousal)


def delivery(valence: float, arousal: float, confidence: float, expressiveness: float = 1.0) -> Tuple[float, float, float]:
    """(rate, pitch, volume): profiles blended by inverse squared distance to each emotion's (valence, arousal), then
    scaled by strength and clamped as prosody_profiles does. At a canonical point it equals that emotion's profile."""
    weights = {l: 1 / ((valence - cv) ** 2 + (arousal - ca) ** 2 + 1e-9) for l, (cv, ca) in CANONICAL_VA.items()}
    total = sum(weights.values())
    blended = [sum(w * PROFILES[l][k] for l, w in weights.items()) / total for k in range(3)]
    s = strength(expressiveness, valence, arousal, confidence)
    return tuple(max(lo, min(hi, 1 + (m - 1) * s)) for m, (lo, hi) in zip(blended, CLAMPS.values()))


def describe(plan) -> List[str]:
    """Table lines for a plan's segments (used by this tool and by run_pipeline)."""
    lines = [f"{'#':<3}{'source':<11}{'emotion':<11}{'valence':>8}{'arousal':>8}   rate  pitch  volume   clause"]
    for k, s in enumerate(plan.segments, start=1):
        p = s.prosody
        lines.append(f"{k:<3}{s.source:<11}{s.emotion_label:<11}{s.valence:>+8.2f}{s.arousal:>+8.2f}"
                     f"   {p.rate:.2f}  {p.pitch:.2f}   {p.volume:.2f}    {s.text}")
    return lines


def main(argv=None) -> None:
    from dataclasses import replace

    from interpreter.contracts import Prosody, SpeechSegment
    from interpreter.kg.demo import demo_graph
    from interpreter.kg.provider import KGContext
    from interpreter.task1 import Task1Config, plan_speech
    from interpreter.testset import DEFAULT_TESTSET, load_testset, make_row

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--testset", nargs="?", const=str(DEFAULT_TESTSET), metavar="CSV")
    ap.add_argument("--text", help="one sentence; may contain {emotion} markers")
    ap.add_argument("--emotion", default="neutral", choices=EMOTIONS, help="with --text: the sentence emotion")
    ap.add_argument("--confidence", type=float, default=0.8, help="with --text: its confidence")
    ap.add_argument("--only", help="comma-separated row ids")
    ap.add_argument("--carry", action="store_true", help="carry each speaker's state from one sentence to the next")
    ap.add_argument("--render", action="store_true", help="write <id>-single.wav and <id>-flow.wav (needs the venv)")
    ap.add_argument("--play", action="store_true", help="render, then play single then flow for each sentence")
    ap.add_argument("--tts", choices=("kokoro", "indextts", "piper"), help="engine (default: config.TTS_ENGINE)")
    ap.add_argument("--expressiveness", type=float, default=1.0)
    ap.add_argument("--target-lang", default="en-NZ", choices=[l for l in LANGS if l != "zh-CN"])
    ap.add_argument("--out", default=str(HERE / "audio_out" / "flow"))
    args = ap.parse_args(argv)
    if not (args.text or args.testset):
        ap.error("give --text or --testset")

    rows = [make_row("cli", args.text, args.emotion, args.confidence)] if args.text else load_testset(args.testset)
    if args.only:
        rows = [r for r in rows if r.frame.frame_id in set(args.only.split(","))]
    ctx, cfg, state = KGContext(demo_graph()), Task1Config(expressiveness=args.expressiveness), SpeakerState()
    out_dir = Path(args.out) / args.target_lang
    for row in rows:
        f, e = row.frame, row.frame.emotion
        previous = state.get(f.speaker_id) if args.carry else None
        plan = plan_speech(f, ctx, cfg, target_lang=args.target_lang, previous=previous)
        print(f"\n===== {f.frame_id}  sentence emotion {e.label} (confidence {e.confidence:.2f})"
              + (f"  [carried state {previous}]" if previous else ""))
        print(f'  "{row.marked_text}"')
        for line in describe(plan):
            print("  " + line)
        if plan.segments:
            state.update(f.speaker_id, plan.segments[-1].valence, plan.segments[-1].arousal)
        if (args.render or args.play) and plan.segments:
            from interpreter.local_tts.engines import engine_for
            from interpreter.local_tts.render import render_speech
            engine = engine_for(args.target_lang, args.tts)
            r, p, v = delivery(e.valence, e.arousal, e.confidence, args.expressiveness)
            single = replace(plan, segments=[SpeechSegment(plan.utterance, e.label, Prosody(
                r, p, v, emphasis=plan.prosody.emphasis, pauses=plan.prosody.pauses), e.valence, e.arousal)])
            out_dir.mkdir(parents=True, exist_ok=True)
            a = render_speech(single, engine, out_dir / f"{f.frame_id}-single.wav").path
            b = render_speech(plan, engine, out_dir / f"{f.frame_id}-flow.wav").path
            print(f"  single: {a}\n  flow  : {b}")
            if args.play:
                from interpreter.local_tts.playback import play
                print("  playing single...")
                play(a)
                print("  playing flow...")
                play(b)


if __name__ == "__main__":
    main()
