"""Pronunciation hints for kupu Māori that IndexTTS speaks (only for listeners who know the kupu; everyone else hears
the English word). IndexTTS-2.5 accepts `<word|CMU phonemes>` inline, syllables separated by " . ", e.g.
`<minute|M IH1 . N AH0 T>`; it protects these from its text normaliser and reads the phonemes instead of guessing.

Built from published pronunciations (2026-10-19): the IPA in each entry's comment is from Wiktionary's Māori section
(en.wiktionary.org/wiki/<kupu>); mahi and koha have no Māori IPA there, so they follow the regular spelling rules. Sound
mapping, from the Victoria University of Wellington te reo pronunciation guide (short a "as in aloud", long a "as in
car", e "as in entry", i "as in eat", o "as in ordinary", u "as in to", r "a soft rolled r", wh "like the English f"):
    short a = AH    long ā = AA    e = EH    i = IY    o, ō = AO    u = UW    ai = AY    au = AW    wh = F    r = R
CMU phonemes are English sounds, so these are approximations (no tapped r, no vowel length). A te reo speaker or
adviser must check them by ear before any real use. To compare candidate hints for one kupu by ear:

    interpreter/.venv/bin/python -m interpreter.local_tts.indextts_phonemes whare             # current + candidates
    interpreter/.venv/bin/python -m interpreter.local_tts.indextts_phonemes whare "F AH1 . D EH0"
    interpreter/.venv/bin/python -m interpreter.local_tts.indextts_phonemes --changed    # every hint that changed

Each candidate is spoken in the same sentence and played with its number; the files stay in audio_out/kupu/.
"""
import re
from typing import Dict, List, Optional, Tuple

from interpreter.concepts import find_kupu

PHONEMES: Dict[str, str] = {
    "ck:whanau": "F AA1 . N AW0",                      # /ˈɸaːnau/ [ˈfɑːnɐʉ]
    "ck:hui": "HH UW1 . IY0",                          # /ˈhui/ [ˈhʉi]
    "ck:kia-ora": "K IY1 . AH0 . AO1 . R AH0",         # /ˈkia ˈora/ [ˈkiɐ ˈɔɾɐ]
    "ck:kai": "K AY1",                                 # /ˈkai/ [ˈkɐi]
    "ck:mahi": "M AH1 . HH IY0",                       # no Wiktionary IPA; spelling rules: /ˈmahi/
    "ck:tamariki": "T AH1 . M AH0 . D IY0 . K IY0",    # /ˈtamariki/ [ˈtɐmɐɾiki]; D for the tapped r: Jay's pick
    "ck:whare": "F AH1 . R EH0",                       # /ˈɸare/ [ˈfɐɾɛ]
    "ck:korero": "K AO1 . R EH0 . R AO0",              # /ˈkoːrero/ [ˈkoːɾɛɾɔ]
    "ck:koha": "K AO1 . HH AH0",                       # no Wiktionary IPA; spelling rules: /ˈkoha/
}

# Extra candidates for auditioning (the current PHONEMES entry is always candidate 1). The first draft used the long
# "car" vowel (AA) for every a; Jay heard whare as wrong (2026-10-19), and the sources say short a is "as in aloud"
# (AH). The old drafts stay here for comparison, plus D as a stand-in for the tapped r.
CANDIDATES: Dict[str, List[str]] = {
    "ck:whare": ["F AA1 . R EH0", "F AH1 . D EH0"],
    "ck:tamariki": ["T AA1 . M AA0 . R IY0 . K IY0", "T AH1 . M AH0 . R IY0 . K IY0"],
    "ck:kia-ora": ["K IY1 . AA0 . AO1 . R AA0", "K IY1 . AH0 . AO1 . D AH0"],
    "ck:mahi": ["M AA1 . HH IY0"],
    # koha: Jay heard the current hint as "koka" (2026-10-20). Alternatives: the old AA ending, a different o, no syllable dot,
    # and a stressed ending.
    "ck:koha": ["K AO1 . HH AA0", "K OW1 . HH AH0", "K AO1 HH AH0", "K AO1 . HH AH1"],
}
# whare: wrong in every version so far, with or without a hint (2026-10-19 and 20). If the "f" sound for wh is what sounds
# wrong, a "w" start may sound closer to a listener who hears wh as "w"; plus "fo" and the tapped-r stand-in D.
CANDIDATES["ck:whare"] = ["F AA1 . R EH0", "F AH1 . D EH0", "W AA1 . R EH0", "W AH1 . D EH0", "F AO1 . R EH0"]
CHANGED = ("whare", "tamariki", "kia ora", "mahi", "koha")   # the hints that changed from the first draft
# Jay's audition (2026-10-19): tamariki = the D version (now current); whare and kia ora sounded wrong in every version,
# including no hint; mahi and koha unclear. IndexTTS cannot say these well; a te reo voice is needed (see README).
SENTENCE = "We talked about the {kupu} today."      # the kupu mid-sentence, as in the pipeline, to hear any pauses around it


def annotate(text: str) -> str:
    """Wrap every kupu in `text` that has a hint as <kupu|phonemes>; other words are left alone."""
    for slug, word in sorted(set(find_kupu(text)), key=lambda x: -len(x[1])):   # longest first ("kia ora")
        phonemes = PHONEMES.get(slug)
        if phonemes:
            text = re.sub(rf"(?<![<\w]){re.escape(word)}(?![\w|])", f"<{word}|{phonemes}>", text)
    return text


def candidates_for(slug: str, own=()) -> List[Tuple[str, Optional[str]]]:
    """[(label, phonemes or None)]: no hint first, then the current hint, then the alternatives (or `own`), then the current
    hint without its syllable dots (to hear whether the dots cause the pauses around a kupu)."""
    out = [("no hint (IndexTTS guesses)", None)]
    for ph in (own or [PHONEMES[slug], *CANDIDATES.get(slug, [])]):
        out.append((ph + ("   <- current" if ph == PHONEMES[slug] else ""), ph))
    nodots = PHONEMES[slug].replace(" . ", " ")
    if not own and nodots != PHONEMES[slug] and nodots not in [ph for _, ph in out]:
        out.append((nodots + "   (current without dots)", nodots))
    return out


def main(argv=None) -> None:
    import argparse
    from array import array
    from pathlib import Path

    from interpreter.concepts import CONCEPTS, lookup_by_te_reo
    from interpreter.local_tts import config
    from interpreter.local_tts.engines import engine_for
    from interpreter.local_tts.indextts_emotion import ALPHAS, VECTORS
    from interpreter.local_tts.playback import play
    from interpreter.local_tts.render import RenderConfig, _normalise_and_write, trim_silence

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kupu", nargs="?", help="the kupu, e.g. whare")
    ap.add_argument("phonemes", nargs="*", help="your own candidate hints (default: the current one plus alternatives)")
    ap.add_argument("--changed", action="store_true", help=f"audition every changed hint: {', '.join(CHANGED)}")
    ap.add_argument("--sentence", default=SENTENCE, help="carrier sentence with {kupu} where the word goes")
    ap.add_argument("--no-play", action="store_true", help="render only")
    ap.add_argument("--ai-host", help="speak through the AI server at this address (default: the AI_HOST variable)")
    ap.add_argument("--ai-port", type=int, help="AI server port (default: AI_PORT, else 7898)")
    ap.add_argument("--local", action="store_true", help="use IndexTTS on this machine instead of the AI server")
    args = ap.parse_args(argv)
    config.AI_SERVER_HOST = args.ai_host or config.AI_SERVER_HOST
    config.AI_SERVER_PORT = args.ai_port or config.AI_SERVER_PORT
    if not (args.kupu or args.changed):
        ap.error("give a kupu or --changed")

    words = CHANGED if args.changed else [args.kupu]
    slugs = [lookup_by_te_reo(w) for w in words]
    if None in slugs:
        raise SystemExit(f"not in the concept table; known: {', '.join(e.te_reo for e in CONCEPTS.values())}")
    engine = engine_for("en-NZ", "indextts" if args.local or not config.AI_SERVER_HOST else "aiserver")
    print(f"speaking through: {type(engine).__name__}" + (f" at {engine.address}" if hasattr(engine, "address") else ""))
    out = Path(__file__).resolve().parent.parent / "audio_out" / "kupu"
    out.mkdir(parents=True, exist_ok=True)
    for slug in slugs:
        word, cands = CONCEPTS[slug].te_reo, candidates_for(slug, () if args.changed else args.phonemes)
        print(f"\n===== {word}")
        paths = []
        for k, (label, ph) in enumerate(cands):
            pcm, rate = engine.synthesize_emotion(args.sentence.format(kupu=f"<{word}|{ph}>" if ph else word),
                                                  VECTORS["neutral"], ALPHAS["neutral"])
            samples = array("h")
            samples.frombytes(pcm)
            path = out / f"{slug.split(':')[1]}-{k}.wav"
            _normalise_and_write(trim_silence(samples, rate), rate, 1.0, path, RenderConfig())
            paths.append(path)
            print(f"  {k}: {label}")
        if args.no_play:
            continue
        choice = ""
        while choice != "q" and choice != "n":
            if not choice:
                for k, (label, _) in enumerate(cands):
                    print(f"playing {k}: {label}")
                    play(paths[k])
            elif choice.isdigit() and int(choice) < len(paths):
                play(paths[int(choice)])
            choice = input(f"{word}: a number = play it, Enter = all again, n = next kupu, q = quit: ").strip().lower()
        if choice == "q":
            break
    print("\ntell Claude which number sounded right for each kupu")

if __name__ == "__main__":
    main()
