"""NAOqi ALTextToSpeech tagged-string builder (Pepper). String only; nothing is sent to a robot.

NAOqi has no phoneme control, so lexicon `ipa` is ignored and `respelling` is flattened to a
lowercase, hyphen-free spelling.
"""
from typing import List, Optional

from interpreter.contracts import SpeechPlan
from interpreter.tts.base import (Rendered, TTSConfig, add_note, clamp, lexicon_spans, match_emphasis,
                                  pause_map, split_punct, tokens)

_MACRONS = str.maketrans("āēīōūĀĒĪŌŪ", "aeiouAEIOU")


def naoqi_respelling(respelling: str) -> str:
    """Lowercase, hyphens removed. UNVERIFIED how NAOqi reads this."""
    return respelling.lower().replace("-", "")


def render(plan: SpeechPlan, cfg: Optional[TTSConfig] = None) -> Rendered:
    cfg = cfg or TTSConfig()
    notes: List[str] = ["NZ English voice availability unconfirmed; check getAvailableVoices() on the robot"]
    p = plan.prosody

    rate = round(clamp(round(100 * p.rate), 50, 400, notes, "rspd"))
    pitch = round(clamp(round(100 * p.pitch), 50, 200, notes, "vct"))
    vol = round(clamp(round(cfg.naoqi_base_volume * p.volume), 0, 100, notes, "vol"))
    header = f"\\rst\\\\rspd={rate}\\\\vct={pitch}\\\\vol={vol}\\"

    toks = tokens(plan.utterance)
    spans = lexicon_spans(toks, plan.lexicon)
    pauses = pause_map(plan, spans)
    emph = match_emphasis(toks, p.emphasis) if cfg.naoqi_emphasis == "emph_tag" else set()
    if p.emphasis and cfg.naoqi_emphasis == "emph_tag" and emph:
        add_note(notes, "emph tag semantics unverified on Pepper; check in the listening test")
    elif p.emphasis:
        add_note(notes, "emphasis skipped (naoqi_emphasis='none' or word not found)")

    stripped: List[str] = []
    parts, i = [], 0
    while i < len(toks):
        end, entry = spans.get(i, (i + 1, None))
        lead, trail = split_punct(toks[i])[0], split_punct(toks[end - 1])[2]
        if entry is not None:
            core = naoqi_respelling(entry.respelling)
            add_note(notes, "lexicon respellings are lowercased with hyphens removed; "
                            "unverified how NAOqi reads them")
        else:
            text = " ".join(toks[i:end])
            core = text[len(lead):len(text) - len(trail)]
            plain = core.translate(_MACRONS)
            if plain != core:
                stripped.append(f"{core}->{plain}")
                core = plain
        if emph & set(range(i, end)):
            core = f"\\emph=1\\ {core} \\emph=0\\"
        seg = lead + core + trail
        seg += "".join(f" \\pau={ms}\\" for ms in pauses.get(end - 1, []))
        parts.append(seg)
        i = end
    if stripped:
        add_note(notes, "macrons stripped (NAOqi English voice): " + ", ".join(stripped))

    payload = f"{header} {' '.join(parts)} \\rst\\"
    return Rendered("naoqi", payload, {"language": "English"}, tuple(notes))
