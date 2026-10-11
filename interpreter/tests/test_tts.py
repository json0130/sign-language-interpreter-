"""Step 2b-i checks (no network). Run from repo root:  python3 -m unittest discover -s interpreter/tests -t . -v"""
import re
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from interpreter.context import StaticContext
from interpreter.contracts import EyeColour, LexiconEntry, Pause, Prosody, SpeechPlan, Voice
from interpreter.mocks.mock_recogniser import load_frames
from interpreter.task1 import plan_speech
from interpreter.tts import azure, naoqi, polly
from interpreter.tts.base import BACKENDS, TTSConfig, render, render_all

FRAMES = Path(__file__).resolve().parents[1] / "mocks" / "frames"
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
SSML_NS = "{http://www.w3.org/2001/10/synthesis}"
UTT = "My whānau and mātua are coming to the hui, tomorrow."


def plan(utterance=UTT, rate=1.10, pitch=1.05, volume=0.9, emphasis=("whānau",), pauses=((8, 150),),
         lexicon=(("whānau", "FAH-no", None), ("hui", "HOO-ee", None)), **kw) -> SpeechPlan:
    return SpeechPlan(
        frame_id="f-t", mode="interpret", utterance=utterance, emotion_label="excited",
        prosody=Prosody(rate=rate, pitch=pitch, volume=volume, emphasis=list(emphasis),
                        pauses=[Pause(i, ms) for i, ms in pauses]),
        lexicon=[LexiconEntry(w, r, ipa=i) for w, r, i in lexicon], **kw)


def text_of(ssml: str) -> str:
    return " ".join("".join(ET.fromstring(ssml).itertext()).split())


class NaoqiTests(unittest.TestCase):
    def test_tags_words_and_respelling(self):
        r = naoqi.render(plan(), TTSConfig())
        for tag in (r"\rst\ ", r"\rspd=110\ ", r"\vct=105\ ", r"\vol=72\ ", r"\pau=150\ ", r"\emph=1\ ", r"\emph=0\ "):
            self.assertIn(tag.strip(), r.payload)
        self.assertTrue(r.payload.startswith(r"\rst\\rspd=110\\vct=105\\vol=72\ "))
        self.assertTrue(r.payload.endswith(r" \rst\ "[:-1]))
        self.assertIn("fahno", r.payload); self.assertIn("hooee", r.payload)
        self.assertNotIn("whānau", r.payload)
        self.assertIn("matua", r.payload); self.assertNotIn("mātua", r.payload)
        self.assertTrue(any("mātua->matua" in n for n in r.notes))
        self.assertTrue(any("emph tag semantics unverified" in n for n in r.notes))
        self.assertEqual(r.params, {"language": "English"})
        self.assertEqual(r.backend, "naoqi")

    def test_clamps_and_default_cfg(self):
        self.assertIn(r"\rspd=200\ "[:-1], naoqi.render(plan(rate=2.0)).payload)
        r = naoqi.render(plan(volume=1.6), TTSConfig(naoqi_base_volume=80))
        self.assertIn(r"\vol=100\ "[:-1], r.payload)
        self.assertTrue(any("vol clamped" in n for n in r.notes))
        self.assertIn(r"\vct=50\ "[:-1], naoqi.render(plan(pitch=0.5)).payload)

    def test_emphasis_none_and_no_emphasis_words(self):
        r = naoqi.render(plan(), TTSConfig(naoqi_emphasis="none"))
        self.assertNotIn("emph", r.payload)
        self.assertTrue(any("emphasis skipped" in n for n in r.notes))
        self.assertNotIn("emph", naoqi.render(plan(emphasis=())).payload)
        with self.assertRaises(ValueError): TTSConfig(naoqi_emphasis="bold")

    def test_phrase_lexicon_and_words_unchanged(self):
        p = plan("Kia ora, we are here.", emphasis=(), pauses=((0, 100),),
                 lexicon=(("kia ora", "kee-ah OR-ah", None),))
        r = naoqi.render(p)
        self.assertIn("keeah orah, " + r"\pau=100\ ".rstrip(), r.payload)
        plain = re.sub(r"\\[a-z]+(=\d+)?\\", "", naoqi.render(plan(lexicon=(), emphasis=(), pauses=())).payload)
        self.assertEqual(" ".join(plain.split()), UTT.translate(str.maketrans("āēīōū", "aeiou")))


class SsmlTests(unittest.TestCase):
    def test_polly(self):
        r = polly.render(plan())
        root = ET.fromstring(r.payload)
        self.assertEqual(root.tag, "speak")
        pro = root.find("prosody")
        self.assertEqual(pro.get("rate"), "110%")
        self.assertRegex(pro.get("volume"), r"^[+-]\d+\.\ddB$")
        self.assertIsNone(pro.get("pitch"))
        self.assertIsNone(root.find(".//emphasis"))
        self.assertIsNotNone(root.find(".//break"))
        self.assertEqual(root.find(".//break").get("time"), "150ms")
        self.assertTrue(any("emphasis dropped: <emphasis> unavailable on Polly neural voices" in n for n in r.notes))
        self.assertEqual(r.params, {"Engine": "neural", "VoiceId": "Aria", "LanguageCode": "en-NZ", "TextType": "ssml"})
        pitched = ET.fromstring(polly.render(plan(), TTSConfig(polly_use_pitch=True)).payload)
        self.assertEqual(pitched.find("prosody").get("pitch"), "+5%")

    def test_azure(self):
        r = azure.render(plan())
        root = ET.fromstring(r.payload)
        self.assertEqual(root.tag, SSML_NS + "speak")
        self.assertEqual(root.attrib, {"version": "1.0", XML_LANG: "en-NZ"})
        voice = root.find(SSML_NS + "voice")
        self.assertEqual(voice.get("name"), "en-NZ-MitchellNeural")
        pro = voice.find(SSML_NS + "prosody")
        self.assertEqual((pro.get("rate"), pro.get("pitch"), pro.get("volume")), ("1.10", "+5%", "-10%"))
        self.assertIsNone(root.find(".//" + SSML_NS + "emphasis"))
        self.assertTrue(any("documented for only three en-US voices" in n for n in r.notes))
        self.assertEqual(r.params, {"voice": "en-NZ-MitchellNeural"})
        wild = ET.fromstring(azure.render(plan(pitch=2.0, rate=2.0)).payload)
        self.assertEqual(wild.find(f".//{SSML_NS}prosody").get("pitch"), "+50%")
        self.assertEqual(wild.find(f".//{SSML_NS}prosody").get("rate"), "2.00")

    def test_phoneme_only_with_ipa(self):
        for mod, ns in ((polly, ""), (azure, SSML_NS)):
            none = mod.render(plan())
            self.assertNotIn("<phoneme", none.payload)
            self.assertTrue(any("no IPA for 'hui'" in n for n in none.notes))
            with_ipa = mod.render(plan(lexicon=(("hui", "HOO-ee", "TEST-IPA"),)))
            ph = ET.fromstring(with_ipa.payload).find(f".//{ns}phoneme")
            self.assertEqual((ph.get("alphabet"), ph.get("ph"), ph.text), ("ipa", "TEST-IPA", "hui"))
            self.assertEqual(text_of(with_ipa.payload), UTT)

    def test_text_integrity_and_escaping(self):
        nasty = "Tom & Jerry <3 the hui, said \"hi\"."
        for mod in (polly, azure):
            r = mod.render(plan(nasty, emphasis=(), pauses=((3, 200),), lexicon=(("hui", "HOO-ee", "TEST-IPA"),)))
            self.assertEqual(text_of(r.payload), nasty)
            self.assertNotIn("Tom & Jerry", r.payload)
            self.assertEqual(text_of(mod.render(plan()).payload), UTT)

    def test_voice_id_override(self):
        p = plan(voice=Voice(voice_id="Custom-1"))
        self.assertEqual(polly.render(p).params["VoiceId"], "Custom-1")
        self.assertEqual(azure.render(p).params["voice"], "Custom-1")
        self.assertIn('name="Custom-1"', azure.render(p).payload)


class DispatchTests(unittest.TestCase):
    def test_every_mock_frame_renders_on_every_backend(self):
        frames = load_frames(FRAMES, "sign_to_speech")
        self.assertEqual(len(frames), 5)
        modes = set()
        for f in frames:
            p = plan_speech(f, StaticContext()); modes.add(p.mode)
            out = render_all(p)
            self.assertEqual(set(out), set(BACKENDS))
            for b, r in out.items():
                self.assertTrue(r.payload.strip())
                self.assertEqual(r.backend, b)
                if b != "naoqi":
                    ET.fromstring(r.payload)
                    self.assertEqual(text_of(r.payload), p.utterance)
        self.assertEqual(modes, {"interpret", "clarify"})

    def test_cfg_none_and_bad_backend(self):
        p = plan()
        for b in BACKENDS:
            self.assertEqual(render(p, b), render(p, b, TTSConfig()))
        with self.assertRaises(ValueError): render(p, "espeak")


if __name__ == "__main__":
    unittest.main()
