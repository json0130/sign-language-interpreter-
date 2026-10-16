"""A small demo graph matching the mock frames, so the pipeline can run without a real KG.

PLACEHOLDERS: the culture priors below are guesses. Which kupu a given community or person knows needs review by
Māori language advisers or speakers before any real use.
"""
from interpreter.concepts import CONCEPTS
from interpreter.kg.graph import Graph, culture_id


def demo_graph() -> Graph:
    g = Graph()
    for slug, entry in CONCEPTS.items():
        g.ensure("concept", slug, entry.te_reo)
    maori = culture_id("Māori")
    g.ensure("culture", maori, "Māori")
    for slug in CONCEPTS:     # one guessed prior for every kupu in the concept table
        g.link(maori, slug, "culture_prior", 0.95, source="demo-seed")
    # The ids are the ones the mock frames use. Aroha signs; she and Rawiri are tagged Māori. visitor_01 and visitor_02
    # have no tag, so the interpreter glosses for them. staff_01 has been told what whānau means: an explicit
    # per-person edge, independent of any culture tag.
    for pid, name in (("person:aroha", "Aroha"), ("person:rawiri", "Rawiri"), ("person:visitor_01", "Visitor 01"),
                      ("person:visitor_02", "Visitor 02"), ("person:staff_01", "Staff 01")):
        g.ensure("person", pid, name)
    for pid in ("person:aroha", "person:rawiri"):
        g.link(pid, maori, "belongs_to_culture", source="demo-seed")
    g.link("person:staff_01", "ck:whanau", "knows_kupu", 0.8, source="demo-seed")
    return g
