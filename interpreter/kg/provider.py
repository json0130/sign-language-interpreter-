"""KG-backed ContextProvider: answers 'does this listener know this kupu Māori?' from the graph."""
from typing import Tuple

from interpreter.kg.graph import Graph

KNOWS_THRESHOLD = 0.5   # weight at or above this counts as "knows"


class KGContext:
    def __init__(self, graph: Graph, threshold: float = KNOWS_THRESHOLD) -> None:
        self.graph, self.threshold = graph, threshold

    def listener_knows_te_reo(self, listener_id: str, slug: str) -> bool:
        """True if the listener knows the kupu `slug` (a ck: id); `explain` says why."""
        return self.explain(listener_id, slug)[0]

    def explain(self, listener_id: str, slug: str) -> Tuple[bool, str]:
        """(knows, reason). An explicit person -> concept `knows_kupu` edge decides either way. Without one, the
        listener's culture tags decide: any culture whose `culture_prior` for the kupu reaches the threshold. A listener
        who is not in the graph, or has neither, does not know it (so the interpreter says the plain English word)."""
        if listener_id not in self.graph.nodes:
            return False, f"{listener_id} is not in the graph"
        explicit = self.graph.edge(listener_id, slug, "knows_kupu")
        if explicit is not None:
            return explicit.weight >= self.threshold, f"knows_kupu edge, weight {explicit.weight:g}"
        tags = self.graph.out(listener_id, "belongs_to_culture")
        for tag in tags:
            prior = self.graph.edge(tag.target_id, slug, "culture_prior")
            if prior is not None and prior.weight >= self.threshold:
                return True, f"belongs to {tag.target_id}, whose prior for it is {prior.weight:g}"
        if tags:
            return False, "culture tag(s) " + ", ".join(t.target_id for t in tags) + " give no prior at or above the threshold"
        return False, "no knows_kupu edge and no culture tag"
