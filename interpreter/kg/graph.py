"""Small knowledge graph for the interpreter: who people are, which cultures they are tagged with, and which kupu
Māori they (or their culture) are likely to know. Standard library only.

A stdlib port of the idea behind the ChatBox V5 graph_relationship module (deterministic ids, a provenance record on
every edge, replace-on-newer upserts, JSON persistence), cut down to what the pipeline needs.

Node ids:  person:<slug>   culture:<slug>   ck:<slug>   (ck: ids are the keys of interpreter/concepts.py)
Edge kinds (all weights are in [0, 1]):
    belongs_to_culture  person  -> culture   a background TAG, never a fact about the person
    culture_prior       culture -> concept   how likely someone from this culture knows the kupu
    knows_kupu          person  -> concept   how well this person knows the kupu (overrides the culture prior)
"""
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

NODE_KINDS = ("person", "culture", "concept")
EDGE_KINDS = ("belongs_to_culture", "culture_prior", "knows_kupu")
# (source node kind, target node kind) each edge kind must connect
ENDS = {"belongs_to_culture": ("person", "culture"), "culture_prior": ("culture", "concept"),
        "knows_kupu": ("person", "concept")}


def slug(label: str) -> str:
    """Stable ASCII slug: lowercase, macrons folded, dashes between words ('Te Reo Māori' -> 'te-reo-maori')."""
    plain = "".join(c for c in unicodedata.normalize("NFD", str(label).strip().lower()) if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "-", plain).strip("-")


def person_id(name: str) -> str:
    return f"person:{slug(name)}"


def culture_id(label: str) -> str:
    return f"culture:{slug(label)}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Provenance:
    """Who wrote an edge, how sure they were, and when (ISO 8601 UTC, so strings compare in time order)."""
    source: str
    confidence: float = 1.0
    timestamp: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("provenance.source must be non-empty")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"provenance.confidence={self.confidence!r} must be in [0, 1]")


@dataclass
class Node:
    id: str
    kind: str
    label: str = ""
    attrs: dict = field(default_factory=dict)      # e.g. a culture's style_hint

    def __post_init__(self) -> None:
        if self.kind not in NODE_KINDS:
            raise ValueError(f"node kind {self.kind!r} not in {NODE_KINDS}")
        if not self.id.startswith({"person": "person:", "culture": "culture:", "concept": "ck:"}[self.kind]):
            raise ValueError(f"{self.kind} node id {self.id!r} has the wrong prefix")


@dataclass
class Edge:
    source_id: str
    target_id: str
    kind: str
    weight: float
    provenance: Provenance

    def __post_init__(self) -> None:
        if self.kind not in EDGE_KINDS:
            raise ValueError(f"edge kind {self.kind!r} not in {EDGE_KINDS}")
        if not 0 <= self.weight <= 1:
            raise ValueError(f"edge weight={self.weight!r} must be in [0, 1]")


class Graph:
    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[Tuple[str, str, str], Edge] = {}   # keyed (source, target, kind): one edge per triple

    def add_node(self, node: Node) -> Node:
        """Create or replace a node."""
        self.nodes[node.id] = node
        return node

    def ensure(self, kind: str, node_id: str, label: str = "") -> Node:
        """The node with this id, created if missing (an existing node is left as it is)."""
        return self.nodes.get(node_id) or self.add_node(Node(node_id, kind, label))

    def link(self, source_id: str, target_id: str, kind: str, weight: float = 1.0, *,
             source: str = "manual", confidence: float = 1.0, timestamp: Optional[str] = None) -> Edge:
        """Upsert an edge. An older write than the stored one is discarded; otherwise it replaces it."""
        return self.add_edge(Edge(source_id, target_id, kind, weight,
                                  Provenance(source, confidence, timestamp or _now())))

    def add_edge(self, edge: Edge) -> Edge:
        for end, node_id, kind in zip(("source", "target"), (edge.source_id, edge.target_id), ENDS[edge.kind]):
            node = self.nodes.get(node_id)
            if node is None:
                raise ValueError(f"{edge.kind}: {end} {node_id!r} is not in the graph")
            if node.kind != kind:
                raise ValueError(f"{edge.kind}: {end} {node_id!r} is a {node.kind}, expected {kind}")
        key = (edge.source_id, edge.target_id, edge.kind)
        old = self.edges.get(key)
        if old is not None and edge.provenance.timestamp < old.provenance.timestamp:
            return old   # stale write
        self.edges[key] = edge
        return edge

    def edge(self, source_id: str, target_id: str, kind: str) -> Optional[Edge]:
        return self.edges.get((source_id, target_id, kind))

    def out(self, source_id: str, kind: Optional[str] = None) -> List[Edge]:
        # ponytail: linear scan; add a per-node index if the graph grows past a few thousand edges
        return [e for e in self.edges.values() if e.source_id == source_id and (kind is None or e.kind == kind)]

    # ---------------------------------------------------------------- JSON persistence
    def to_dict(self) -> dict:
        return {"nodes": [asdict(n) for n in self.nodes.values()], "edges": [asdict(e) for e in self.edges.values()]}

    @classmethod
    def from_dict(cls, d: dict) -> "Graph":
        g = cls()
        for n in d.get("nodes", []):
            g.add_node(Node(**n))
        for e in d.get("edges", []):
            g.add_edge(Edge(**{**e, "provenance": Provenance(**e["provenance"])}))
        return g

    def save(self, path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path) -> "Graph":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))
