"""Emotion backup for the transcript: trust the recogniser's emotion, fall back to simple text rules when it is weak.

Rules only (no model): first matching rule wins. They are a crude backup for transcripts with little or no emotion
signal, not an emotion recogniser, and they read English wording only.
"""
import re
from dataclasses import replace
from typing import List, Optional, Tuple

from interpreter.contracts import Emotion, EmotionSpan, InterpretationFrame

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


def nearest_label(valence: float, arousal: float) -> str:
    """The emotion whose canonical (valence, arousal) is closest."""
    return min(CANONICAL_VA, key=lambda l: (CANONICAL_VA[l][0] - valence) ** 2 + (CANONICAL_VA[l][1] - arousal) ** 2)


# ---------------------------------------------------------------- grammar vs emotion
# In NZSL, as in many sign languages, the face also does grammar: raised eyebrows can mark a yes/no question and furrowed
# eyebrows a wh-question. A face-reading emotion recogniser can mistake those for emotions. To be confirmed with Deaf
# advisers; starting values.
QUESTION_WORDS = ("what", "where", "which", "who", "whom", "whose", "when", "why", "how")
LOOKALIKES = {"yes/no": ("surprised",), "wh": ("angry", "confused")}   # question type -> readings that may be grammar
GRAMMAR_DISCOUNT = 0.4     # such a reading is set aside as neutral, keeping this share of its confidence. Voicing a calm
                           # question as angry is worse than missing anger in a question; the words can still add
                           # emotion through the text rules (e.g. "Wow, really?").


def question_type(text: str) -> Optional[str]:
    """"wh" or "yes/no" if `text` is a question (ends with ? and has no !), else None."""
    t = text.strip()
    if not t.endswith("?") or "!" in t:
        return None
    return "wh" if set(re.findall(r"[a-z']+", t.lower())) & set(QUESTION_WORDS) else "yes/no"


def _discount(e: Emotion) -> Emotion:
    return Emotion("neutral", 0.0, 0.0, round(e.confidence * GRAMMAR_DISCOUNT, 3))


def guard_grammar(frame: InterpretationFrame) -> Tuple[InterpretationFrame, List[str]]:
    """Discount face readings that a question's eyebrow grammar can produce: on the sentence emotion when the whole
    sentence is a question, and on each emotion_flow span whose own words are a question. Returns (frame, notes)."""
    notes: List[str] = []
    emotion, spans = frame.emotion, list(frame.emotion_flow)
    q = question_type(frame.text)
    if q and emotion.label in LOOKALIKES[q]:
        new = _discount(emotion)
        notes.append(f"{q} question: the {emotion.label} reading may be eyebrow grammar, not emotion; set aside -> "
                     f"neutral (confidence {emotion.confidence:.2f} -> {new.confidence:.2f})")
        emotion = new
    words = frame.text.split()
    for i, span in enumerate(spans):
        end = spans[i + 1].start_word if i + 1 < len(spans) else len(words)
        q = question_type(" ".join(words[span.start_word:end]))
        if q and span.emotion.label in LOOKALIKES[q]:
            new = _discount(span.emotion)
            notes.append(f"from word {span.start_word}: {q} question, the {span.emotion.label} reading may be eyebrow "
                         f"grammar; set aside -> neutral (confidence {span.emotion.confidence:.2f} -> "
                         f"{new.confidence:.2f})")
            spans[i] = EmotionSpan(span.start_word, new)
    if not notes:
        return frame, notes
    return replace(frame, emotion=emotion, emotion_flow=spans), notes


def prepare_emotion(frame: InterpretationFrame) -> Tuple[InterpretationFrame, List[str]]:
    """The emotion stage: the grammar guard, then the text-rule backup for weak readings. Returns (frame, guard notes)."""
    guarded, notes = guard_grammar(frame)
    return refine_emotion(guarded), notes
