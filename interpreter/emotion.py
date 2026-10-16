"""Emotion backup for the transcript: trust the recogniser's emotion, fall back to simple text rules when it is weak.

Rules only (no model): first matching rule wins. They are a crude backup for transcripts with little or no emotion
signal, not an emotion recogniser, and they read English wording only.
"""
import re
from dataclasses import replace
from typing import Optional, Tuple

from interpreter.contracts import Emotion, InterpretationFrame

# Canonical (valence, arousal) per emotion, in [-1, 1] (Russell circumplex style). Starting values, not measurements.
CANONICAL_VA = {
    "neutral": (0.0, 0.0), "happy": (0.7, 0.4), "excited": (0.7, 0.8), "calm": (0.4, -0.5),
    "sad": (-0.7, -0.4), "apologetic": (-0.3, -0.1), "angry": (-0.7, 0.7), "anxious": (-0.5, 0.6),
    "surprised": (0.3, 0.8), "confused": (-0.2, 0.2),
}
RULE_CONFIDENCE = 0.5      # how sure a text rule is; deliberately middling
MIN_CONFIDENCE = 0.4       # the recogniser's own emotion is kept at or above this

# (label, pattern), checked in order. Order matters: strong/negative cues first, 'sorry' last because it is common.
RULES = [
    ("angry", r"\b(angry|furious|unacceptable|ridiculous|outrageous|hate)\b"),
    ("anxious", r"\b(worried|nervous|afraid|scared|anxious|running late|missed)\b"),
    ("sad", r"\b(sad|lonely|grief|grieving|heartbroken|miss(ed|ing)? (my|you|him|her|them|home))\b"),
    ("excited", r"\b(excited|can't wait|over the moon|amazing|awesome|yay)\b|!!"),
    ("happy", r"\b(thanks?|thank you|great|wonderful|glad|happy|pleased|lov(e|ed|es|ing)|enjoy(ed|ing)?)\b"),
    ("surprised", r"\b(wow|no way|unexpected|surprised|can't believe)\b"),
    ("confused", r"\b(confused|don't understand|what do you mean|not sure what)\b"),
    ("apologetic", r"\b(sorry|apologi[sz]e|excuse me|pardon)\b"),
]


def match_rule(text: str) -> Optional[Tuple[str, str]]:
    """(label, the text that matched) for the first rule that matches `text`, or None."""
    for label, pattern in RULES:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            return label, m.group(0)
    return None


def guess_emotion(text: str) -> Optional[str]:
    """The emotion label of the first rule that matches `text`, or None."""
    found = match_rule(text)
    return found[0] if found else None


def refine_emotion(frame: InterpretationFrame) -> InterpretationFrame:
    """The frame with a usable emotion: unchanged when the recogniser's confidence is at least MIN_CONFIDENCE,
    otherwise the text-rule guess (confidence RULE_CONFIDENCE) if a rule matches, otherwise unchanged."""
    if frame.emotion.confidence >= MIN_CONFIDENCE:
        return frame
    label = guess_emotion(frame.text)
    if label is None:
        return frame
    valence, arousal = CANONICAL_VA[label]
    return replace(frame, emotion=Emotion(label, valence, arousal, RULE_CONFIDENCE))
