"""Pronunciation hints for kupu Māori that IndexTTS speaks (only for listeners who know the kupu; everyone else hears
the English word). IndexTTS-2.5 accepts `<word|CMU phonemes>` inline, syllables separated by " . ", e.g.
`<minute|M IH1 . N AH0 T>`; it protects these from its text normaliser and reads the phonemes instead of guessing.

DRAFTS, NOT VERIFIED: these are approximations of te reo sounds in English (ARPAbet) phonemes, written without a te reo
speaker. CMU phonemes cannot express everything: the tapped r has no symbol (R is used), long vowels are only marked by
stress, and "wh" is approximated as F. A te reo speaker or adviser must check them by ear before any real use.
Roughly:  a = AA, e = EH, i = IY, o = AO, u = UW, au = AW, ai = AY.
"""
import re
from typing import Dict

from interpreter.concepts import find_kupu

PHONEMES: Dict[str, str] = {
    "ck:whanau": "F AA1 . N AW0",                      # whā-nau
    "ck:hui": "HH UW1 . IY0",                          # hu-i
    "ck:kia-ora": "K IY1 . AA0 . AO1 . R AA0",         # ki-a o-ra
    "ck:kai": "K AY1",                                 # kai
    "ck:mahi": "M AA1 . HH IY0",                       # ma-hi
    "ck:tamariki": "T AA1 . M AA0 . R IY0 . K IY0",    # ta-ma-ri-ki
    "ck:whare": "F AA1 . R EH0",                       # wha-re
    "ck:korero": "K AO1 . R EH0 . R AO0",              # kō-re-ro
    "ck:koha": "K AO1 . HH AA0",                       # ko-ha
}


def annotate(text: str) -> str:
    """Wrap every kupu in `text` that has a hint as <kupu|phonemes>; other words are left alone."""
    for slug, word in sorted(set(find_kupu(text)), key=lambda x: -len(x[1])):   # longest first ("kia ora")
        phonemes = PHONEMES.get(slug)
        if phonemes:
            text = re.sub(rf"(?<![<\w]){re.escape(word)}(?![\w|])", f"<{word}|{phonemes}>", text)
    return text
