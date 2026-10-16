"""Test sentences from one CSV file, turned into frames for the pipeline and the emotion-flow tool.

Columns (header row required; only id, text and emotion must be filled):
    id          frame id, e.g. t01
    mix         free label for grouping, e.g. "english" or "english+kupu"
    text        the transcript. Optional {emotion} markers start a new clause emotion for the flow tool, e.g.
                "{happy}Kia ora everyone, {sad}but koro could not come." The pipeline strips them.
    emotion     the recogniser's utterance emotion (one of contracts.EMOTIONS)
    confidence  the recogniser's emotion confidence, 0..1 (default 0.8). Below 0.4 the text rules take over.
    listener    addressee id in the KG (default person:visitor_01, who knows no te reo)
    concepts    space-separated ck: slugs the recogniser tagged, e.g. "ck:whanau ck:hui" (lets an English word be
                spoken as te reo for a listener who knows it)
    notes       what to listen for
    speaker, input_confidence, valence, arousal   optional overrides
"""
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

from interpreter.concepts import CONCEPTS, gloss
from interpreter.contracts import Concept, Emotion, InterpretationFrame
from interpreter.emotion import CANONICAL_VA

DEFAULT_TESTSET = Path(__file__).resolve().parent / "mocks" / "testset.csv"
MARKER = re.compile(r"\{([a-z]+)\}\s*")


def strip_markers(text: str) -> str:
    return MARKER.sub("", text).strip()


@dataclass(frozen=True)
class TestRow:
    frame: InterpretationFrame     # what the pipeline gets (markers removed)
    marked_text: str               # the text as written, markers included (what the flow tool reads)
    mix: str = ""
    notes: str = ""


def make_row(row_id: str, text: str, emotion: str = "neutral", confidence: float = 0.8,
             listener: str = "person:visitor_01", concepts: str = "", speaker: str = "person:aroha",
             input_confidence: float = 0.9, valence=None, arousal=None, mix: str = "", notes: str = "") -> TestRow:
    v, a = CANONICAL_VA.get(emotion, (0.0, 0.0))
    tags = []
    for slug in concepts.split():
        if slug not in CONCEPTS:
            raise ValueError(f"unknown concept {slug!r}; known: {', '.join(CONCEPTS)}")
        tags.append(Concept(slug, gloss(slug, "en-NZ"), CONCEPTS[slug].te_reo))
    frame = InterpretationFrame(
        frame_id=row_id, direction="sign_to_speech", speaker_id=speaker, addressee_id=listener,
        text=strip_markers(text), input_confidence=float(input_confidence), concepts=tags,
        emotion=Emotion(emotion, float(v if valence in (None, "") else valence),
                        float(a if arousal in (None, "") else arousal), float(confidence)))
    return TestRow(frame, text, mix, notes)


def load_testset(path=DEFAULT_TESTSET) -> List[TestRow]:
    """Every row of the CSV as a TestRow. Errors name the row."""
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for n, r in enumerate(csv.DictReader(fh), start=2):
            fields = {k.strip(): (v or "").strip() for k, v in r.items() if k}
            if not fields.get("id"):
                continue                     # blank line
            try:
                kwargs = {k: v for k, v in fields.items() if v != "" and k != "id"}
                rows.append(make_row(fields["id"], **kwargs))
            except (ValueError, TypeError) as e:
                raise ValueError(f"{Path(path).name} line {n} ({fields.get('id')}): {e}") from e
    ids = [r.frame.frame_id for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{Path(path).name}: duplicate ids")
    return rows
