import json
import unittest
from pathlib import Path

from interpreter.concepts import CONCEPTS, find_kupu, gloss, lookup_by_te_reo
from interpreter.context import StaticContext
from interpreter.contracts import (LANGS, SCHEMA_VERSION, SUPPORTED_SCHEMAS, Concept, InterpretationFrame,
                                   SignPlan, SpeechPlan, Voice)
from interpreter.mocks.mock_recogniser import load_frames
from interpreter.task1 import plan_speech
from interpreter.tests.test_contracts import frame_d1, frame_d2

FRAMES = Path(__file__).resolve().parent.parent / "mocks" / "frames"


def new_style(**kw) -> dict:
    d = frame_d1()
    d["text"] = d.pop("text_en")
    d.update(kw)
    return d


class TestFrameLanguage(unittest.TestCase):
    def test_legacy_mock_frames_load_and_expose_text(self):
        raw = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(FRAMES.glob("*.json"))]
        self.assertTrue(all("text_en" in d for d in raw))   # the files are still in the old format
        frames = load_frames(FRAMES)
        self.assertEqual(len(frames), len(raw))
        for f, d in zip(frames, raw):
            self.assertEqual(f.text, d["text_en"])
            self.assertEqual((f.source_lang, f.target_lang), ("en-NZ", "en-NZ"))

    def test_new_style_roundtrip(self):
        f = InterpretationFrame.from_dict(new_style(source_lang="en-US", target_lang="en-NZ"))
        out = f.to_dict()
        self.assertIn("text", out)
        self.assertNotIn("text_en", out)
        f2 = InterpretationFrame.from_dict(json.loads(f.to_json()))
        self.assertEqual(f, f2)
        self.assertEqual((f2.source_lang, f2.target_lang), ("en-US", "en-NZ"))

    def test_text_en_property_is_read_only_alias(self):
        f = InterpretationFrame.from_dict(frame_d1())
        self.assertEqual(f.text_en, f.text)
        with self.assertRaises(AttributeError):
            f.text_en = "x"

    def test_text_wins_when_both_keys_present(self):
        d = new_style(); d["text_en"] = "old"
        self.assertNotEqual(InterpretationFrame.from_dict(d).text, "old")

    def test_schema_01_accepted_and_output_is_02(self):
        self.assertEqual(SUPPORTED_SCHEMAS, ("0.1", "0.2"))
        d = frame_d1(); d["schema_version"] = "0.1"
        f = InterpretationFrame.from_dict(d)
        self.assertEqual(f.schema_version, SCHEMA_VERSION)
        self.assertEqual(f.to_dict()["schema_version"], "0.2")
        for cls, d in ((SpeechPlan, {"frame_id": "a", "mode": "interpret", "utterance": "Hi.",
                                     "emotion_label": "neutral", "schema_version": "0.1"}),
                       (SignPlan, {"frame_id": "a", "meaning": "Hi.", "units": [{"concept": "greeting"}],
                                   "emotion_target": {"label": "neutral", "valence": 0, "arousal": 0},
                                   "schema_version": "0.1"})):
            self.assertEqual(cls.from_dict(d).schema_version, "0.2")
        d = frame_d1(); d["schema_version"] = "0.3"
        with self.assertRaises(ValueError):
            InterpretationFrame.from_dict(d)

    def test_invalid_languages_rejected(self):
        for key in ("source_lang", "target_lang"):
            with self.assertRaises(ValueError):
                InterpretationFrame.from_dict(new_style(**{key: "fr"}))
        for lang in LANGS:
            Voice(language=lang)
        with self.assertRaises(ValueError):
            Voice(language="fr")
        with self.assertRaises(ValueError):
            Concept("ck:x", "x", glosses={"fr": "x"})
        with self.assertRaises(ValueError):
            Concept("ck:x", "x", glosses={"en-NZ": "  "})

    def test_text_to_speech_direction(self):
        d = frame_d2(); d["direction"] = "text_to_speech"
        self.assertEqual(InterpretationFrame.from_dict(d).direction, "text_to_speech")
        d["gloss"] = "X"
        with self.assertRaises(ValueError):   # gloss rule unchanged
            InterpretationFrame.from_dict(d)

    def test_concept_glosses_roundtrip(self):
        c = Concept("ck:hui", "meeting", "hui", glosses={"en-NZ": "meeting", "zh-CN": "会议"})
        self.assertEqual(Concept.from_dict(json.loads(c.to_json())), c)
        self.assertEqual(Concept("ck:hui", "meeting").glosses, {})


class TestConcepts(unittest.TestCase):
    def test_lookup_is_macron_and_case_insensitive(self):
        for w in ("whānau", "whanau", "Whānau", "WHANAU", " whānau "):
            self.assertEqual(lookup_by_te_reo(w), "ck:whanau", w)
        self.assertEqual(lookup_by_te_reo("kia ora"), "ck:kia-ora")
        self.assertIsNone(lookup_by_te_reo("family"))

    def test_find_kupu_handles_macrons_case_and_phrases(self):
        found = find_kupu("Kia ora! My WHANAU and my whānau are at the hui, not the whanaunga.")
        self.assertEqual(found, [("ck:whanau", "WHANAU"), ("ck:whanau", "whānau"), ("ck:hui", "hui"),
                                 ("ck:kia-ora", "Kia ora")])

    def test_gloss(self):
        self.assertEqual(gloss("ck:whanau", "en-US"), "family")
        self.assertEqual(gloss("ck:hui", "en-NZ"), "meeting")
        self.assertIsNone(gloss("ck:whanau", "zh-CN"))
        self.assertIsNone(gloss("ck:nope", "en-NZ"))

    def test_entries_flagged_for_review_and_zh_empty(self):
        for slug, e in CONCEPTS.items():
            self.assertTrue(e.needs_review, slug)
            self.assertNotIn("zh-CN", e.glosses, slug)
            self.assertIn("fluent speakers", e.note, slug)


class TestPlanSpeechLanguage(unittest.TestCase):
    def setUp(self):
        self.ctx = StaticContext()
        self.frame = InterpretationFrame.from_dict(frame_d1())

    def test_voice_language_follows_target(self):
        self.assertEqual(plan_speech(self.frame, self.ctx).voice.language, "en-NZ")
        self.assertEqual(plan_speech(self.frame, self.ctx, target_lang="en-US").voice.language, "en-US")
        self.frame.target_lang = "en-US"
        self.assertEqual(plan_speech(self.frame, self.ctx).voice.language, "en-US")
        self.assertEqual(plan_speech(self.frame, self.ctx, target_lang="en-NZ").voice.language, "en-NZ")

    def test_en_nz_and_en_us_plan_the_same_words_and_prosody(self):
        a, b = (plan_speech(self.frame, self.ctx, target_lang=l) for l in ("en-NZ", "en-US"))
        self.assertEqual((a.utterance, a.prosody, a.eye), (b.utterance, b.prosody, b.eye))

    def test_clarify_plan_carries_language(self):
        self.frame.input_confidence = 0.2
        p = plan_speech(self.frame, self.ctx, target_lang="en-US")
        self.assertEqual((p.mode, p.voice.language), ("clarify", "en-US"))

    def test_text_to_speech_accepted(self):
        d = frame_d2(); d["direction"] = "text_to_speech"
        self.assertEqual(plan_speech(InterpretationFrame.from_dict(d), self.ctx).mode, "interpret")

    def test_speech_to_sign_rejected(self):
        with self.assertRaises(ValueError):
            plan_speech(InterpretationFrame.from_dict(frame_d2()), self.ctx)

    def test_zh_cn_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            plan_speech(self.frame, self.ctx, target_lang="zh-CN")
        self.frame.target_lang = "zh-CN"
        with self.assertRaises(NotImplementedError):
            plan_speech(self.frame, self.ctx)

    def test_unknown_language_rejected(self):
        with self.assertRaises(ValueError):
            plan_speech(self.frame, self.ctx, target_lang="fr")


if __name__ == "__main__":
    unittest.main()
