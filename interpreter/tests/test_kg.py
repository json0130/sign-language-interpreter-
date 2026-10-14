import json
import tempfile
import unittest
from pathlib import Path

from interpreter.kg.demo import demo_graph
from interpreter.kg.graph import Edge, Graph, Node, Provenance, culture_id, person_id, slug
from interpreter.kg.provider import KGContext


def small() -> Graph:
    g = Graph()
    g.ensure("person", "person:a", "A")
    g.ensure("culture", "culture:maori", "Māori")
    g.ensure("concept", "ck:whanau", "whānau")
    return g


class TestGraph(unittest.TestCase):
    def test_ids(self):
        self.assertEqual(slug("Te Reo!"), "te-reo")
        self.assertEqual(slug("Te Reo Māori"), "te-reo-maori")
        self.assertEqual(person_id("Aroha"), "person:aroha")
        self.assertEqual(culture_id("Māori"), "culture:maori")

    def test_node_validation(self):
        with self.assertRaises(ValueError):
            Node("person:a", "robot")
        with self.assertRaises(ValueError):
            Node("culture:x", "person")   # wrong prefix for the kind

    def test_edge_validation(self):
        g = small()
        with self.assertRaises(ValueError):
            g.link("person:a", "culture:maori", "friend_of")                      # unknown kind
        with self.assertRaises(ValueError):
            g.link("person:a", "culture:nope", "belongs_to_culture")              # missing endpoint
        with self.assertRaises(ValueError):
            g.link("person:a", "ck:whanau", "belongs_to_culture")                 # wrong target kind
        with self.assertRaises(ValueError):
            g.link("person:a", "ck:whanau", "knows_kupu", 1.5)                    # weight out of range
        with self.assertRaises(ValueError):
            Provenance("", 1.0)
        with self.assertRaises(ValueError):
            Provenance("x", 2.0)

    def test_replace_on_newer_and_discard_stale(self):
        g = small()
        g.link("person:a", "ck:whanau", "knows_kupu", 0.2, timestamp="2026-01-01T00:00:00+00:00")
        g.link("person:a", "ck:whanau", "knows_kupu", 0.9, timestamp="2026-02-01T00:00:00+00:00")
        self.assertEqual(g.edge("person:a", "ck:whanau", "knows_kupu").weight, 0.9)
        g.link("person:a", "ck:whanau", "knows_kupu", 0.1, timestamp="2026-01-15T00:00:00+00:00")   # older: ignored
        self.assertEqual(g.edge("person:a", "ck:whanau", "knows_kupu").weight, 0.9)
        self.assertEqual(len(g.edges), 1)

    def test_ensure_does_not_overwrite(self):
        g = small()
        g.nodes["person:a"].attrs["note"] = "kept"
        g.ensure("person", "person:a", "other label")
        self.assertEqual(g.nodes["person:a"].attrs, {"note": "kept"})

    def test_json_roundtrip(self):
        g = demo_graph()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "kg.json"
            g.save(path)
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(set(data), {"nodes", "edges"})
            self.assertIn("Māori", path.read_text(encoding="utf-8"))   # not escaped to ä
            g2 = Graph.load(path)
        self.assertEqual(g.to_dict(), g2.to_dict())


class TestKGContext(unittest.TestCase):
    def setUp(self):
        self.g = small()
        self.ctx = KGContext(self.g)

    def test_unknown_listener_does_not_know(self):
        self.assertFalse(self.ctx.listener_knows_te_reo("person:nobody", "ck:whanau"))
        self.assertFalse(self.ctx.listener_knows_te_reo("person:a", "ck:whanau"))   # in the graph, no tag

    def test_culture_prior_counts_only_at_threshold(self):
        self.g.link("person:a", "culture:maori", "belongs_to_culture")
        self.g.link("culture:maori", "ck:whanau", "culture_prior", 0.4)
        self.assertFalse(self.ctx.listener_knows_te_reo("person:a", "ck:whanau"))
        self.g.link("culture:maori", "ck:whanau", "culture_prior", 0.5)
        self.assertTrue(self.ctx.listener_knows_te_reo("person:a", "ck:whanau"))

    def test_explicit_edge_beats_culture_prior_both_ways(self):
        self.g.link("person:a", "culture:maori", "belongs_to_culture")
        self.g.link("culture:maori", "ck:whanau", "culture_prior", 0.95)
        self.g.link("person:a", "ck:whanau", "knows_kupu", 0.1)       # this person does not know it
        self.assertFalse(self.ctx.listener_knows_te_reo("person:a", "ck:whanau"))
        self.g.ensure("person", "person:b", "B")
        self.g.link("person:b", "ck:whanau", "knows_kupu", 0.9)       # no culture tag, but knows it
        self.assertTrue(self.ctx.listener_knows_te_reo("person:b", "ck:whanau"))

    def test_demo_graph_matches_the_mock_frames(self):
        ctx = KGContext(demo_graph())
        self.assertTrue(ctx.listener_knows_te_reo("person:aroha", "ck:hui"))
        self.assertFalse(ctx.listener_knows_te_reo("person:visitor_01", "ck:whanau"))
        self.assertTrue(ctx.listener_knows_te_reo("person:staff_01", "ck:whanau"))
        self.assertFalse(ctx.listener_knows_te_reo("person:staff_01", "ck:hui"))


if __name__ == "__main__":
    unittest.main()
