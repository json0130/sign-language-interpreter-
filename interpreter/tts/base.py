"""Shared pieces for the TTS adapters. String builders only: no network, no SDKs, no API keys.

Adapters must not change what is said: apart from lexicon substitution and macron stripping on
NAOqi, the words in a payload equal the words in `plan.utterance`.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from xml.sax.saxutils import escape

from interpreter.contracts import LexiconEntry, SpeechPlan

BACKENDS = ("naoqi", "polly", "azure")
_PUNCT = ".,!?;:"


@dataclass(frozen=True)
class TTSConfig:
    naoqi_base_volume: int = 80          # placeholder, calibrate on the robot
    naoqi_emphasis: str = "emph_tag"     # "emph_tag" or "none"
    polly_use_pitch: bool = False
    polly_voice: str = "Aria"
    azure_voice: str = "en-NZ-MitchellNeural"

    def __post_init__(self) -> None:
        if self.naoqi_emphasis not in ("emph_tag", "none"):
            raise ValueError(f"naoqi_emphasis={self.naoqi_emphasis!r} must be 'emph_tag' or 'none'")


@dataclass(frozen=True)
class Rendered:
    backend: str
    payload: str
    params: dict
    notes: Tuple[str, ...]


# ---------------------------------------------------------------- text helpers
def tokens(utterance: str) -> List[str]:
    return utterance.split()


def strip_punct(token: str) -> str:
    return token.strip(_PUNCT)


def split_punct(token: str) -> Tuple[str, str, str]:
    """(leading punctuation, core, trailing punctuation)."""
    core = strip_punct(token)
    if not core:
        return token, "", ""
    start = token.index(core)
    return token[:start], core, token[start + len(core):]


def match_emphasis(toks: List[str], emphasis_words: List[str]) -> set:
    """Indices of tokens that carry an emphasis word (case-insensitive, punctuation ignored)."""
    wanted = {strip_punct(w).casefold() for w in emphasis_words}
    return {i for i, t in enumerate(toks) if strip_punct(t).casefold() in wanted}


def escape_xml(text: str) -> str:
    """Safe for both element text and double-quoted attributes."""
    return escape(text, {'"': "&quot;"})


def clamp(value: float, lo: float, hi: float, notes: List[str], label: str) -> float:
    if value < lo or value > hi:
        clipped = max(lo, min(hi, value))
        note = f"{label} clamped from {value:g} to {clipped:g}"
        if note not in notes:
            notes.append(note)
        return clipped
    return value


def add_note(notes: List[str], note: str) -> None:
    if note not in notes:
        notes.append(note)


# ---------------------------------------------------------------- lexicon spans and pauses
def lexicon_spans(toks: List[str], lexicon: List[LexiconEntry]) -> Dict[int, Tuple[int, LexiconEntry]]:
    """{start: (end_exclusive, entry)} for each lexicon word or phrase found in the tokens."""
    keys = [strip_punct(t).casefold() for t in toks]
    spans: Dict[int, Tuple[int, LexiconEntry]] = {}
    covered: set = set()
    for e in lexicon:
        want = [strip_punct(w).casefold() for w in e.word.split()]
        n = len(want)
        for i in range(len(toks) - n + 1):
            if keys[i:i + n] == want and not covered & set(range(i, i + n)):
                spans[i] = (i + n, e)
                covered.update(range(i, i + n))
    return spans


def pause_map(plan: SpeechPlan, spans: Dict[int, Tuple[int, LexiconEntry]]) -> Dict[int, List[int]]:
    """{token index to pause after: [ms, ...]}; a pause inside a phrase moves to the phrase end."""
    out: Dict[int, List[int]] = {}
    for p in plan.prosody.pauses:
        idx = p.after_word_index
        for start, (end, _) in spans.items():
            if start <= idx < end:
                idx = end - 1
        out.setdefault(idx, []).append(p.ms)
    return out


def ssml_body(plan: SpeechPlan, notes: List[str]) -> str:
    """Escaped text with <break/> pauses and <phoneme> for lexicon entries that have IPA.
    Shared by Polly and Azure. Without IPA a lexicon word is left as-is and noted."""
    toks = tokens(plan.utterance)
    spans = lexicon_spans(toks, plan.lexicon)
    pauses = pause_map(plan, spans)
    parts, i = [], 0
    while i < len(toks):
        end, entry = spans.get(i, (i + 1, None))
        lead, trail = split_punct(toks[i])[0], split_punct(toks[end - 1])[2]
        text = " ".join(toks[i:end])
        if entry is not None and entry.ipa:
            core = text[len(lead):len(text) - len(trail)]
            seg = (f'{escape_xml(lead)}<phoneme alphabet="ipa" ph="{escape_xml(entry.ipa)}">'
                   f'{escape_xml(core)}</phoneme>{escape_xml(trail)}')
        else:
            if entry is not None:
                add_note(notes, f"no IPA for {entry.word!r}; left unannotated (needs a te reo speaker or adviser)")
            seg = escape_xml(text)
        seg += "".join(f'<break time="{ms}ms"/>' for ms in pauses.get(end - 1, []))
        parts.append(seg)
        i = end
    return " ".join(parts)


# ---------------------------------------------------------------- dispatch
def render(plan: SpeechPlan, backend: str, cfg: Optional[TTSConfig] = None) -> Rendered:
    from interpreter.tts import azure, naoqi, polly   # late import: the backends import this module
    modules = {"naoqi": naoqi, "polly": polly, "azure": azure}
    if backend not in modules:
        raise ValueError(f"backend={backend!r} not in {BACKENDS}")
    return modules[backend].render(plan, cfg or TTSConfig())


def render_all(plan: SpeechPlan, cfg: Optional[TTSConfig] = None) -> Dict[str, Rendered]:
    return {b: render(plan, b, cfg) for b in BACKENDS}
