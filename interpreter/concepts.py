"""Concept table: ck: slug -> te reo form and plain-word glosses per language.

PLACEHOLDERS: every entry needs review by a te reo speaker or adviser before any real use.
zh-CN glosses are left empty on purpose; they must be supplied and reviewed by fluent speakers.
"""
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, Optional

_ZH_NOTE = "to be supplied and reviewed by fluent speakers"


@dataclass(frozen=True)
class ConceptEntry:
    te_reo: str
    glosses: Dict[str, str] = field(default_factory=dict)   # lang (contracts.LANGS) -> text
    needs_review: bool = True
    note: str = ""


CONCEPTS: Dict[str, ConceptEntry] = {
    "ck:whanau": ConceptEntry("whānau", {"en-NZ": "family", "en-US": "family"}, note=f"zh-CN gloss {_ZH_NOTE}"),
    "ck:hui": ConceptEntry("hui", {"en-NZ": "meeting", "en-US": "meeting"}, note=f"zh-CN gloss {_ZH_NOTE}"),
    "ck:kia-ora": ConceptEntry("kia ora", {"en-NZ": "hello", "en-US": "hello"}, note=f"zh-CN gloss {_ZH_NOTE}"),
}


def _fold(word: str) -> str:
    """Case- and macron-insensitive key: 'Whānau' -> 'whanau'."""
    decomposed = unicodedata.normalize("NFD", word.strip().casefold())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


_BY_TE_REO = {_fold(e.te_reo): slug for slug, e in CONCEPTS.items()}


def lookup_by_te_reo(word: str) -> Optional[str]:
    """The ck: slug for a kupu Māori ('whanau' and 'Whānau' both give 'ck:whanau'), or None."""
    return _BY_TE_REO.get(_fold(word))


def gloss(slug: str, lang: str) -> Optional[str]:
    """Plain-word rendering of a concept in `lang`, or None if the concept or language is missing."""
    entry = CONCEPTS.get(slug)
    return entry.glosses.get(lang) if entry else None
