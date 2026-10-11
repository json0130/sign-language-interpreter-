"""Pronunciation respellings for the TTS, te reo word -> NZ English respelling.

PLACEHOLDERS: validate every entry with Māori language advisers or speakers before any real use.
"""
from typing import Optional

LEXICON = {
    "whānau": "FAH-no",
    "hui": "HOO-ee",
    "kia ora": "kee-ah OR-ah",
}


def lookup(word: str) -> Optional[str]:
    """Respelling for `word`, case-insensitive, or None if unknown."""
    return LEXICON.get(word.strip().casefold())


# IPA left empty on purpose. Values must come from a te reo Māori speaker or adviser, not from us.
_IPA: dict = {}


def lookup_ipa(word: str) -> Optional[str]:
    """IPA for `word`, case-insensitive, or None if unknown."""
    return _IPA.get(word.strip().casefold())
