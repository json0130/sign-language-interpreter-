"""Emotion flow: one emotion per clause instead of one per utterance. STANDALONE for now (not wired into the pipeline),
so it can be listened to and tuned first.

    python3 -m interpreter.emotion_flow --testset [CSV] [--only t12,t13]          # print the flow per sentence
    python3 -m interpreter.emotion_flow --text "{happy}Kia ora, {sad}but koro could not come." [--emotion E --confidence C]
    interpreter/.venv/bin/python -m interpreter.emotion_flow --testset --only t12 --play   # A/B: single emotion, then flow
      [--render] [--carry] [--expressiveness 0..2] [--target-lang en-NZ|en-US] [--out DIR]

How a clause gets its emotion, in order:
  1. an authored {emotion} marker (confidence 0.9), used as written;
  2. otherwise a text rule (emotion.py) matching the clause, blended with the utterance emotion by their confidences;
  3. otherwise the utterance emotion (the recogniser's).
Rule- and utterance-driven clauses keep 30% of the previous clause (inertia), so the voice does not jump. With --carry, the
first clause of a sentence also keeps 30% of that speaker's state at the end of their previous sentence.
Emotion is held as valence/arousal; delivery (rate, pitch, volume) is a distance-weighted blend of the emotion profiles,
so in-between emotions get in-between delivery. Starting values to tune by ear.
"""
import argparse
import array
import re
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from interpreter.contracts import EMOTIONS, Emotion, LANGS, Pause, Prosody, SpeechPlan, Voice
from interpreter.emotion import CANONICAL_VA, RULE_CONFIDENCE, match_rule
from interpreter.prosody_profiles import CLAMPS, PROFILES, strength

HERE = Path(__file__).resolve().parent
MARKER = re.compile(r"\{([a-z]+)\}")
# break after sentence punctuation, or before a contrast/sequence joiner that follows a comma
CLAUSE_BREAK = re.compile(r"(?<=[.!?;])\s+|(?<=,)\s+(?=(?:but|and then|so|although|though)\b)", re.IGNORECASE)
MARKER_CONFIDENCE = 0.9
INERTIA = 0.3        # share of the previous clause kept by a rule- or utterance-driven clause
STATE_CARRY = 0.3    # share of the speaker's previous state kept by the first clause (with --carry)
GAP_MS = 120         # silence between clauses when they are joined into one WAV


@dataclass(frozen=True)
class Segment:
    text: str
    valence: float
    arousal: float
    confidence: float
    source: str          # "marker", "rule" or "utterance"

    @property
    def label(self) -> str:
        return nearest_label(self.valence, self.arousal)


def nearest_label(valence: float, arousal: float) -> str:
    return min(CANONICAL_VA, key=lambda l: (CANONICAL_VA[l][0] - valence) ** 2 + (CANONICAL_VA[l][1] - arousal) ** 2)


def clauses(marked_text: str) -> List[Tuple[Optional[str], str]]:
    """[(marker label or None, clause text)]. A marker applies to every clause up to the next marker."""
    parts = MARKER.split(marked_text)               # [text, label, text, label, text, ...]
    chunks = [(None, parts[0])] + list(zip(parts[1::2], parts[2::2]))
    out = []
    for label, chunk in chunks:
        if label is not None and label not in EMOTIONS:
            raise ValueError(f"unknown emotion marker {{{label}}}; use one of {', '.join(EMOTIONS)}")
        out += [(label, c.strip()) for c in CLAUSE_BREAK.split(chunk) if c and c.strip()]
    return out


def _clamp(x: float) -> float:
    return max(-1.0, min(1.0, x))


def flow(marked_text: str, utterance: Emotion, previous: Optional[Tuple[float, float]] = None) -> List[Segment]:
    """Per-clause emotions for one utterance. `previous` is the speaker's last (valence, arousal), for carry-over."""
    segments: List[Segment] = []
    for i, (label, text) in enumerate(clauses(marked_text)):
        if label is not None:
            (v, a), conf, source = CANONICAL_VA[label], MARKER_CONFIDENCE, "marker"
        else:
            rule = match_rule(text)
            if rule:
                rv, ra = CANONICAL_VA[rule[0]]
                cu, cr = utterance.confidence, RULE_CONFIDENCE
                v = (cu * utterance.valence + cr * rv) / (cu + cr)
                a = (cu * utterance.arousal + cr * ra) / (cu + cr)
                conf, source = max(cu, cr), "rule"
            else:
                v, a, conf, source = utterance.valence, utterance.arousal, utterance.confidence, "utterance"
            anchor = (segments[-1].valence, segments[-1].arousal) if segments else (previous if i == 0 else None)
            if anchor is not None:
                keep = INERTIA if segments else STATE_CARRY
                v, a = (1 - keep) * v + keep * anchor[0], (1 - keep) * a + keep * anchor[1]
        segments.append(Segment(text, round(_clamp(v), 3), round(_clamp(a), 3), conf, source))
    return segments


class SpeakerState:
    """Each speaker's (valence, arousal) at the end of their last utterance."""

    def __init__(self) -> None:
        self.last: Dict[str, Tuple[float, float]] = {}

    def get(self, speaker_id: str) -> Optional[Tuple[float, float]]:
        return self.last.get(speaker_id)

    def update(self, speaker_id: str, segments: List[Segment]) -> None:
        if segments:
            self.last[speaker_id] = (segments[-1].valence, segments[-1].arousal)


def delivery(valence: float, arousal: float, confidence: float, expressiveness: float = 1.0) -> Tuple[float, float, float]:
    """(rate, pitch, volume): profiles blended by inverse squared distance to each emotion's (valence, arousal),
    then scaled by strength and clamped exactly as Task 1 does. At a canonical point it equals that emotion's profile."""
    weights = {l: 1 / ((valence - cv) ** 2 + (arousal - ca) ** 2 + 1e-9) for l, (cv, ca) in CANONICAL_VA.items()}
    total = sum(weights.values())
    blended = [sum(w * PROFILES[l][k] for l, w in weights.items()) / total for k in range(3)]
    s = strength(expressiveness, valence, arousal, confidence)
    return tuple(max(lo, min(hi, 1 + (m - 1) * s)) for m, (lo, hi) in zip(blended, CLAMPS.values()))


def plans(row_id: str, segments: List[Segment], lang: str = "en-NZ", expressiveness: float = 1.0) -> List[SpeechPlan]:
    """One SpeechPlan per clause (same comma pauses and pronunciation lexicon as Task 1)."""
    from interpreter.task1 import lexicon_for
    out = []
    for k, seg in enumerate(segments, start=1):
        rate, pitch, volume = delivery(seg.valence, seg.arousal, seg.confidence, expressiveness)
        tokens = seg.text.split()
        pauses = [Pause(i, 150 if seg.arousal >= 0 else 300) for i, t in enumerate(tokens[:-1]) if t.endswith(",")]
        out.append(SpeechPlan(frame_id=f"{row_id}-{k}", mode="interpret", utterance=seg.text, emotion_label=seg.label,
                              voice=Voice(language=lang), prosody=Prosody(rate, pitch, volume, pauses=pauses),
                              lexicon=lexicon_for(seg.text)))
    return out


# ---------------------------------------------------------------- audio: render clauses and join them into one WAV
def _read(path) -> Tuple[array.array, int]:
    with wave.open(str(path), "rb") as w:
        s = array.array("h")
        s.frombytes(w.readframes(w.getnframes()))
        return s, w.getframerate()


def resample(samples: array.array, src: int, dst: int) -> array.array:
    """Linear-interpolation resample. Each clause is written at its own rate (that is how pitch is applied), so clauses
    are brought to one rate before joining; playback sounds the same as the separate files."""
    if src == dst or not samples:
        return array.array("h", samples)
    n, step, last = round(len(samples) * dst / src), src / dst, len(samples) - 1
    out = array.array("h", bytes(2 * n))
    for i in range(n):
        x = i * step
        j = min(int(x), last)
        nxt = samples[min(j + 1, last)]
        out[i] = round(samples[j] + (nxt - samples[j]) * (x - j))
    return out


def join_wavs(paths, out_path, gap_ms: int = GAP_MS) -> float:
    """Concatenate WAVs (any sample rates) with short silences; returns seconds."""
    loaded = [_read(p) for p in paths]
    rate = max(r for _, r in loaded)
    joined = array.array("h")
    for k, (s, r) in enumerate(loaded):
        if k:
            joined.extend(array.array("h", bytes(2 * round(gap_ms / 1000 * rate))))
        joined.extend(resample(s, r, rate))
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(joined.tobytes())
    return len(joined) / rate


def render(row_id: str, segments: List[Segment], out_dir: Path, lang: str, expressiveness: float, engine=None) -> Path:
    """Render each clause, then join them into <out_dir>/<row_id>.wav (clauses kept in <out_dir>/parts/)."""
    from interpreter.local_tts.render import render_speech
    parts = out_dir / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    paths = [render_speech(p, engine, parts / f"{p.frame_id}.wav").path for p in plans(row_id, segments, lang, expressiveness)]
    out = out_dir / f"{row_id}.wav"
    join_wavs(paths, out)
    return out


def describe(segments: List[Segment], expressiveness: float) -> None:
    print(f"  {'#':<3}{'source':<10}{'emotion':<11}{'valence':>8}{'arousal':>8}{'conf':>6}   rate  pitch  volume   clause")
    for k, s in enumerate(segments, start=1):
        r, p, v = delivery(s.valence, s.arousal, s.confidence, expressiveness)
        print(f"  {k:<3}{s.source:<10}{s.label:<11}{s.valence:>+8.2f}{s.arousal:>+8.2f}{s.confidence:>6.2f}"
              f"   {r:.2f}  {p:.2f}   {v:.2f}    {s.text}")


def main(argv=None) -> None:
    from interpreter.testset import DEFAULT_TESTSET, load_testset, make_row
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--testset", nargs="?", const=str(DEFAULT_TESTSET), metavar="CSV")
    ap.add_argument("--text", help="one sentence; may contain {emotion} markers")
    ap.add_argument("--emotion", default="neutral", choices=EMOTIONS, help="with --text: the utterance emotion")
    ap.add_argument("--confidence", type=float, default=0.8, help="with --text: its confidence")
    ap.add_argument("--only", help="comma-separated row ids")
    ap.add_argument("--carry", action="store_true", help="carry each speaker's state from one sentence to the next")
    ap.add_argument("--render", action="store_true", help="write <id>-single.wav and <id>-flow.wav (needs the venv)")
    ap.add_argument("--play", action="store_true", help="render, then play single then flow for each sentence")
    ap.add_argument("--expressiveness", type=float, default=1.0)
    ap.add_argument("--target-lang", default="en-NZ", choices=[l for l in LANGS if l != "zh-CN"])
    ap.add_argument("--out", default=str(HERE / "audio_out" / "flow"))
    args = ap.parse_args(argv)
    if not (args.text or args.testset):
        ap.error("give --text or --testset")

    rows = ([make_row("cli", args.text, args.emotion, args.confidence)] if args.text else load_testset(args.testset))
    if args.only:
        rows = [r for r in rows if r.frame.frame_id in set(args.only.split(","))]
    state = SpeakerState()
    out_dir = Path(args.out) / args.target_lang
    for row in rows:
        f, e = row.frame, row.frame.emotion
        segs = flow(row.marked_text, e, state.get(f.speaker_id) if args.carry else None)
        print(f"\n===== {f.frame_id}  utterance emotion {e.label} (confidence {e.confidence:.2f})"
              + (f"  [carried state {state.get(f.speaker_id)}]" if args.carry and state.get(f.speaker_id) else ""))
        print(f'  "{row.marked_text}"')
        describe(segs, args.expressiveness)
        state.update(f.speaker_id, segs)
        if args.render or args.play:
            single = [Segment(f.text, e.valence, e.arousal, e.confidence, "utterance")]
            a = render(f"{f.frame_id}-single", single, out_dir, args.target_lang, args.expressiveness)
            b = render(f"{f.frame_id}-flow", segs, out_dir, args.target_lang, args.expressiveness)
            print(f"  single: {a}\n  flow  : {b}")
            if args.play:
                from interpreter.local_tts.playback import play
                print("  playing single...")
                play(a)
                print("  playing flow...")
                play(b)


if __name__ == "__main__":
    main()
