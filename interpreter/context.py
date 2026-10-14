"""What the interpreter knows about the people in the conversation.

Task 1 only ever talks to this interface. interpreter/kg/provider.py is the KG-backed implementation;
StaticContext is a dict-backed stand-in for tests and quick runs.
"""
from typing import Dict, Optional, Protocol, Tuple


class ContextProvider(Protocol):
    def listener_knows_te_reo(self, listener_id: str, slug: str) -> bool:
        """True if the listener (the addressee) knows the kupu Māori for concept `slug` (a ck: id), so it can be
        spoken as te reo. False means say the plain English word instead."""
        ...


class StaticContext:
    """Dict-backed provider: {(listener_id, slug): bool}, falling back to `default`.

    The default is True ("everyone knows the kupu"), which keeps te reo in the speech; use default=False for the
    gloss-everything case.
    """

    def __init__(self, table: Optional[Dict[Tuple[str, str], bool]] = None, default: bool = True) -> None:
        self.table = dict(table or {})
        self.default = default

    def listener_knows_te_reo(self, listener_id: str, slug: str) -> bool:
        return self.table.get((listener_id, slug), self.default)
