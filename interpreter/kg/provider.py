"""KG-backed ContextProvider: answers 'does this listener know this kupu Māori?' from the graph."""
from interpreter.kg.graph import Graph

KNOWS_THRESHOLD = 0.5   # weight at or above this counts as "knows"


class KGContext:
    def __init__(self, graph: Graph, threshold: float = KNOWS_THRESHOLD) -> None:
        self.graph, self.threshold = graph, threshold

    def listener_knows_te_reo(self, listener_id: str, slug: str) -> bool:
        """True if the listener knows the kupu `slug` (a ck: id).

        An explicit person -> concept `knows_kupu` edge decides either way. Without one, the listener's culture tag
        decides: any culture they belong to whose `culture_prior` for the kupu reaches the threshold. A listener who
        is not in the graph, or has neither, does not know it (so the interpreter says the plain English word).
        """
        explicit = self.graph.edge(listener_id, slug, "knows_kupu")
        if explicit is not None:
            return explicit.weight >= self.threshold
        for tag in self.graph.out(listener_id, "belongs_to_culture"):
            prior = self.graph.edge(tag.target_id, slug, "culture_prior")
            if prior is not None and prior.weight >= self.threshold:
                return True
        return False
