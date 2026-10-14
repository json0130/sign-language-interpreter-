"""Task 1 (sign_to_speech or text_to_speech): rule-based speech and expression planner.

Interpret, never reply: the plan carries the SIGNER's words and the SIGNER's emotion. Pepper adds no
content; its own voice is used only for clarification. Rules only: no KG, PAD, LLM or TTS here.
"""
import re
from dataclasses import dataclass
from typing import Optional

from interpreter.concepts import find_kupu, gloss
from interpreter.context import ContextProvider
from interpreter.contracts import (LANGS, EyeColour, InterpretationFrame, LexiconEntry, Pause, Prosody,
                                   SpeechPlan, Voice)
from interpreter.lexicon import LEXICON, lookup, lookup_ipa
from interpreter.prosody_profiles import scaled, strength

# PLACEHOLDERS: colour-emotion associations vary between people and cultures; check them in the
# evaluation step before relying on them.
EYE_PALETTE = {
    "neutral": (200, 220, 255), "happy": (255, 200, 60), "excited": (255, 140, 40),
    "calm": (80, 180, 220), "sad": (50, 80, 200), "apologetic": (150, 120, 200),
    "angry": (220, 40, 40), "anxious": (220, 200, 60), "surprised": (180, 255, 255),
    "confused": (170, 130, 220),
}
CLARIFY_UTTERANCE = "Sorry, one moment, I didn't catch that."
CLARIFY_EYE = EyeColour(rgb=(255, 170, 0), intensity=0.9, pattern="pulse", transition_ms=200)
_PUNCT = ".,!?;:"


@dataclass(frozen=True)
class Task1Config:
    expressiveness: float = 1.0      # 0 = flat delivery, 1 = full profile strength, up to 2 = exaggerated
    clarify_threshold: float = 0.5   # input_confidence below this -> ask for a repeat


def _choose_words(frame: InterpretationFrame, ctx: ContextProvider, lang: str) -> list:
    """(from, to) whole-word swaps so each concept is spoken in the form the LISTENER knows.

    Listener knows the kupu: a tagged concept's English word becomes the te reo form.
    Listener does not: any known kupu in the text (tagged or not, 'whanau' and 'whānau' alike) becomes its plain
    word for `lang`, from interpreter/concepts.py.
    """
    listener = frame.addressee_id
    swaps = [(c.english, c.te_reo) for c in frame.concepts
             if c.te_reo and ctx.listener_knows_te_reo(listener, c.slug)]
    for slug, word in find_kupu(frame.text):
        plain = gloss(slug, lang)
        if plain and not ctx.listener_knows_te_reo(listener, slug):
            swaps.append((word, plain))
    return swaps


def _swap(text: str, pairs: list) -> str:
    """Whole-word, case-insensitive replace, keeping a leading capital."""
    for old, new in pairs:
        def repl(m: re.Match, new: str = new) -> str:
            return new[0].upper() + new[1:] if m.group(0)[0].isupper() else new
        text = re.sub(rf"\b{re.escape(old)}\b", repl, text, flags=re.IGNORECASE)
    return text


def plan_speech(frame: InterpretationFrame, ctx: ContextProvider,
                cfg: Optional[Task1Config] = None, target_lang: Optional[str] = None) -> SpeechPlan:
    """Plan how Pepper speaks and shows the signer's emotion.

    Clarify: if input_confidence < cfg.clarify_threshold, return the robot-voice line plus an amber
    pulse. The pulse is a visual "please repeat" cue for the signer (a spoken request would not reach
    them); the spoken line is for the non-signer. The colour meaning must be validated with Deaf users.

    Word choice follows the LISTENER (frame.addressee_id): te reo is spoken only where ctx says they know the
    kupu, otherwise the plain English word ('whānau' becomes 'family'), whichever form the transcript used.
    Known limitation: no grammar rewriting, so "family is" becomes "whānau is".

    Prosody: the emotion's profile (prosody_profiles.PROFILES) scaled by
    expressiveness x intensity x confidence (see prosody_profiles.strength). Eye colour comes
    from EYE_PALETTE.

    Language: speaks in `target_lang or frame.target_lang`. en-NZ and en-US plan identically and set
    Voice.language; zh-CN is reserved and raises NotImplementedError.
    """
    cfg = cfg or Task1Config()
    if frame.direction not in ("sign_to_speech", "text_to_speech"):
        raise ValueError(f"Task 1 handles sign_to_speech and text_to_speech only, got {frame.direction!r}")
    lang = target_lang or frame.target_lang
    if lang not in LANGS:
        raise ValueError(f"target_lang={lang!r} not in {LANGS}")
    if lang == "zh-CN":
        raise NotImplementedError("Mandarin rendering is not implemented yet")

    if frame.input_confidence < cfg.clarify_threshold:
        return SpeechPlan(frame_id=frame.frame_id, mode="clarify", utterance=CLARIFY_UTTERANCE,
                          emotion_label="neutral", voice=Voice(persona="robot", language=lang), eye=CLARIFY_EYE)

    pairs = _choose_words(frame, ctx, lang)
    utterance = _swap(frame.text, pairs)
    tokens = utterance.split()
    stripped = [t.strip(_PUNCT).casefold() for t in tokens]

    emo = frame.emotion
    rate, pitch, volume = scaled(emo.label, strength(cfg.expressiveness, emo.valence, emo.arousal,
                                                      emo.confidence))

    emphasis = []
    for n in frame.nuance:
        if n.kind != "emphasis":
            continue
        target = _swap(n.target, pairs).strip(_PUNCT)
        if target.casefold() in stripped:   # single-token targets only; the contract can't stress phrases
            word = tokens[stripped.index(target.casefold())].strip(_PUNCT)
            if word not in emphasis:
                emphasis.append(word)

    pauses = [Pause(i, 150 if emo.arousal >= 0 else 300) for i, t in enumerate(tokens) if t.endswith(",")]
    prosody = Prosody(rate=rate, pitch=pitch, volume=volume, emphasis=emphasis, pauses=pauses)

    lively = emo.arousal > 0.5
    eye = EyeColour(rgb=EYE_PALETTE[emo.label], intensity=0.5 + 0.4 * (emo.arousal + 1) / 2,
                    pattern="pulse" if lively else "solid", transition_ms=200 if lively else 400)

    lexicon = [LexiconEntry(w, lookup(w), ipa=lookup_ipa(w)) for w in LEXICON
               if re.search(rf"\b{re.escape(w)}\b", utterance, flags=re.IGNORECASE)]

    return SpeechPlan(frame_id=frame.frame_id, mode="interpret", utterance=utterance,
                      emotion_label=emo.label, voice=Voice(language=lang), prosody=prosody, eye=eye, lexicon=lexicon)
