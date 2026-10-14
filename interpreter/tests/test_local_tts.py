import importlib.util
import math
import tempfile
import unittest
import wave
from array import array
from dataclasses import replace
from pathlib import Path

from interpreter.context import StaticContext
from unittest import mock

from interpreter.contracts import LexiconEntry, Pause, Prosody, SpeechPlan, Voice
from interpreter.local_tts.config import DEFAULT_VOICE, MODELS_DIR, VOICES, RenderConfig
from interpreter.local_tts.piper_engine import engine_for
from interpreter.local_tts.render import render_speech
from interpreter.mocks.mock_recogniser import load_frames
from interpreter.task1 import plan_speech

FRAMES = Path(__file__).resolve().parent.parent / "mocks" / "frames"


class FakeEngine:
    """Deterministic tone; 10 ms of audio (at length_scale 1) per non-space character."""
    native_sample_rate = 16000
    base_length_scale = 0.9

    def __init__(self):
        self.calls = []

    def synthesize(self, text, length_scale):
        self.calls.append((text, length_scale))
        n = round(len(text.replace(" ", "")) * 160 * length_scale)
        return array("h", (round(9000 * math.sin(2 * math.pi * 440 * i / 16000)) for i in range(n))).tobytes()


def plan(utterance="hello there my friend", **kw):
    lexicon = kw.pop("lexicon", [])
    return SpeechPlan(frame_id="t", mode="interpret", utterance=utterance, emotion_label="neutral",
                      prosody=Prosody(**kw), lexicon=lexicon)


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        s = array("h")
        s.frombytes(w.readframes(w.getnframes()))
        return w.getnchannels(), w.getsampwidth(), w.getframerate(), s


def rms(s):
    return math.sqrt(sum(x * x for x in s) / len(s))


class TestRender(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "o.wav"
        self.addCleanup(self.tmp.cleanup)

    def render(self, p, engine=None, cfg=None):
        engine = engine or FakeEngine()
        return engine, render_speech(p, engine, self.out, cfg)

    def test_length_scale_is_base_times_pitch_over_rate(self):
        for r, p in ((1.0, 1.0), (1.2, 1.0), (0.8, 1.1), (1.5, 0.9), (0.6, 1.25)):
            eng, _ = self.render(plan(rate=r, pitch=p))
            self.assertAlmostEqual(eng.calls[0][1], 0.9 * p / r, places=9, msg=(r, p))

    def test_wav_format_and_sample_rate(self):
        for p in (1.0, 1.1, 0.85):
            _, res = self.render(plan(pitch=p))
            ch, width, rate, samples = read_wav(self.out)
            self.assertEqual((ch, width), (1, 2))
            self.assertEqual(rate, round(16000 * p))
            self.assertEqual(res.sample_rate, rate)
            self.assertGreater(len(samples), 0)

    def test_net_tempo_is_rate(self):
        _, base = self.render(plan(rate=1.0, pitch=1.2))
        _, fast = self.render(plan(rate=1.25, pitch=1.2))
        self.assertAlmostEqual(base.duration_s / fast.duration_s, 1.25, places=2)

    def test_pitch_is_clamped_with_note(self):
        _, res = self.render(plan(pitch=2.0))
        self.assertEqual(res.sample_rate, round(16000 * 1.25))
        self.assertTrue(any("pitch clamped" in n for n in res.notes))

    def test_volume_gain(self):
        self.render(plan(volume=1.0)); full = rms(read_wav(self.out)[3])
        self.render(plan(volume=0.5)); half = rms(read_wav(self.out)[3])
        self.assertAlmostEqual(half / full, 0.5, delta=0.02)
        self.render(plan(volume=1.5)); loud = read_wav(self.out)[3]
        self.assertLessEqual(max(abs(x) for x in loud), 32767)
        self.assertGreater(max(abs(x) for x in loud), 0.7 * 32767)

    def test_pause_adds_real_playback_time(self):
        words = "one two three four"
        for pitch in (1.0, 1.2, 0.85):
            _, plain = self.render(plan(words, pitch=pitch))
            _, paused = self.render(plan(words, pitch=pitch, pauses=[Pause(1, 300)]))
            self.assertAlmostEqual(paused.duration_s - plain.duration_s, 0.3, delta=0.02, msg=pitch)

    def test_pause_splits_segments(self):
        eng, _ = self.render(plan("one two three four", pauses=[Pause(1, 200)]))
        self.assertEqual([t for t, _ in eng.calls], ["one two", "three four"])

    def test_lexicon_respelling_both_styles(self):
        lex = [LexiconEntry("whānau", "FAH-no"), LexiconEntry("hui", "HOO-ee")]
        p = plan("My whānau is coming to the hui.", lexicon=lex)
        for style, want in (("hyphen", "My fah-no is coming to the hoo-ee."),
                            ("joined", "My fahno is coming to the hooee.")):
            eng, _ = self.render(p, cfg=RenderConfig(respell_style=style))
            self.assertEqual([t for t, _ in eng.calls], [want])

    def test_no_lexicon_words_unchanged(self):
        eng, _ = self.render(plan("Nothing special here."))
        self.assertEqual(eng.calls[0][0], "Nothing special here.")

    def test_flat_and_expressive_send_same_words(self):
        ctx = StaticContext()
        for f in load_frames(FRAMES, "sign_to_speech"):
            p = plan_speech(f, ctx)
            a, _ = self.render(p)
            b, _ = self.render(replace(p, prosody=Prosody()))
            self.assertEqual(" ".join(t for t, _ in a.calls), " ".join(t for t, _ in b.calls), f.frame_id)

    def test_emphasis_note(self):
        _, res = self.render(plan(emphasis=["friend"]))
        self.assertIn("emphasis not rendered by local backend", res.notes)


class TestEngineByLanguage(unittest.TestCase):
    def test_render_picks_engine_from_plan_language(self):
        class Rate(FakeEngine):
            def __init__(self, rate):
                super().__init__()
                self.native_sample_rate = rate
        engines = {"en-NZ": Rate(16000), "en-US": Rate(22050)}
        with tempfile.TemporaryDirectory() as d, mock.patch(
                "interpreter.local_tts.render.engine_for", side_effect=lambda lang: engines[lang]):
            for lang in engines:
                p = replace(plan(), voice=Voice(language=lang))
                res = render_speech(p, None, Path(d) / "x.wav")
                self.assertEqual(res.sample_rate, engines[lang].native_sample_rate)
                self.assertEqual(len(engines[lang].calls), 1)

    def test_explicit_engine_wins(self):
        eng = FakeEngine()
        with tempfile.TemporaryDirectory() as d, mock.patch("interpreter.local_tts.render.engine_for") as m:
            render_speech(replace(plan(), voice=Voice(language="en-US")), eng, Path(d) / "x.wav")
        m.assert_not_called()
        self.assertEqual(len(eng.calls), 1)

    def test_zh_cn_and_unknown_language(self):
        with self.assertRaises(NotImplementedError):
            engine_for("zh-CN")
        with self.assertRaises(ValueError):
            engine_for("fr")

    def test_registry_has_en_nz_and_en_us(self):
        self.assertEqual(set(VOICES), {"en-NZ", "en-US"})
        self.assertTrue(VOICES["en-US"].startswith("en_US-"))


MODEL = MODELS_DIR / f"{DEFAULT_VOICE}.onnx"
US_MODEL = MODELS_DIR / f"{VOICES['en-US']}.onnx"


@unittest.skipUnless(importlib.util.find_spec("piper") and MODEL.exists(),
                     "piper or the voice model is not available (run with interpreter/.venv)")
class TestRealPiper(unittest.TestCase):
    def test_renders_short_sentence(self):
        from interpreter.local_tts.piper_engine import PiperEngine
        with tempfile.TemporaryDirectory() as d:
            res = render_speech(plan("Hello there.", rate=1.1, pitch=1.05), PiperEngine(MODEL), Path(d) / "x.wav")
            self.assertGreater(res.duration_s, 0)
            self.assertEqual(read_wav(res.path)[0:2], (1, 2))


@unittest.skipUnless(importlib.util.find_spec("piper") and US_MODEL.exists(),
                     "piper or the en-US voice model is not available (run with interpreter/.venv)")
class TestRealPiperEnUS(unittest.TestCase):
    def test_renders_en_us_sentence_by_language(self):
        p = replace(plan("Hello there, how are you today?", rate=1.05), voice=Voice(language="en-US"))
        with tempfile.TemporaryDirectory() as d:
            res = render_speech(p, None, Path(d) / "x.wav")   # engine chosen from plan.voice.language
            self.assertGreater(res.duration_s, 0)
            self.assertEqual(res.params["native_sample_rate"], engine_for("en-US").native_sample_rate)


if __name__ == "__main__":
    unittest.main()
