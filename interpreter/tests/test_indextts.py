import io
import json
import math
import tempfile
import threading
import unittest
import wave
from array import array
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

from interpreter.contracts import Emotion, InterpretationFrame
from interpreter.emotion import CANONICAL_VA
from interpreter.emotion_flow import markers_to_spans
from interpreter.kg.demo import demo_graph
from interpreter.kg.provider import KGContext
from interpreter.local_tts.indextts_emotion import ALPHAS, FLOW_STRENGTH, SLOTS, VECTORS, describe, emotion_vector
from interpreter.local_tts.indextts_phonemes import PHONEMES, annotate
from interpreter.concepts import CONCEPTS
from interpreter.local_tts.indextts_engine import IndexTTSEngine
from interpreter.local_tts import config
from interpreter.local_tts.render import render_speech, trim_silence
from interpreter.run_pipeline import clause_line
from interpreter.task1 import plan_speech


class FakeIndexTTS:
    """Speaks the indextts_server protocol: /health, and /synthesize returning a tone WAV; records every request."""

    def __init__(self, fail: bool = False):
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps({"ok": True, "sample_rate": 22050, "reference": "ref.wav"}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(req)
                if fail:
                    body = b"ValueError: boom"
                    self.send_response(400)
                else:
                    n = 2205 * len(req["text"].split())          # 0.1 s per word, with 0.2 s silence each side
                    tone = array("h", bytes(2 * 4410)) + array("h", (round(8000 * math.sin(i / 10)) for i in range(n))) \
                        + array("h", bytes(2 * 4410))
                    buf = io.BytesIO()
                    with wave.open(buf, "wb") as w:
                        w.setnchannels(1)
                        w.setsampwidth(2)
                        w.setframerate(22050)
                        w.writeframes(tone.tobytes())
                    body = buf.getvalue()
                    self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class TestEmotionVector(unittest.TestCase):
    def test_every_emotion_has_a_vector_and_alpha(self):
        self.assertEqual(set(VECTORS), set(CANONICAL_VA))
        self.assertEqual(set(ALPHAS), set(CANONICAL_VA))
        self.assertTrue(all(len(v) == len(SLOTS) == 8 for v in VECTORS.values()))

    def test_exact_at_each_emotion(self):
        for label, (v, a) in CANONICAL_VA.items():
            vec, alpha = emotion_vector(v, a)
            self.assertEqual((vec, alpha), (VECTORS[label], ALPHAS[label]), label)

    def test_jays_strengths(self):
        self.assertEqual({k: ALPHAS[k] for k in ("neutral", "sad", "excited", "angry")},
                         {"neutral": 0.5, "sad": 0.7, "excited": 0.5, "angry": 0.7})

    def test_between_emotions_mixes(self):
        vec, alpha = emotion_vector(0.0, 0.3)
        self.assertGreater(sum(1 for x in vec if x > 0.01), 2)
        self.assertTrue(0.5 <= alpha <= 0.8)

    def test_describe(self):
        self.assertEqual(describe(VECTORS["sad"]), "sad 0.80, melancholic 0.20")
        self.assertEqual(describe([0] * 8), "none")


class TestDefaultsAndTrim(unittest.TestCase):
    def test_indextts_is_the_default_on_a_port_clear_of_the_chatbox_viewer(self):
        self.assertEqual(config.TTS_ENGINE, "indextts")
        self.assertNotIn(":8765", config.INDEXTTS_URL)

    def test_trim_silence(self):
        sound = array("h", [0] * 1000 + [5000, -5000] * 500 + [0] * 1000)
        out = trim_silence(sound, 10000, pad_ms=10)     # 10 ms = 100 samples of padding each side
        self.assertEqual(len(out), 1000 + 200)
        self.assertEqual(trim_silence(array("h", [0] * 50), 10000), array("h", [0] * 50))


class TestServerChunking(unittest.TestCase):
    def test_hints_are_never_split(self):
        from interpreter.local_tts.indextts_server import chunk_text
        for text in ("My whānau are coming to the hui tomorrow.", "Thank you for the koha, it means a lot to our whānau.",
                     "The tamariki are tired after a long day of mahi.", "Kia ora, thanks for the kai, it was delicious!"):
            for chunk in chunk_text(annotate(text)):
                self.assertEqual(chunk.count("<"), chunk.count(">"), chunk)
            self.assertEqual(" ".join(chunk_text(annotate(text))), annotate(text))

    def test_splits_at_punctuation_within_the_budget(self):
        from interpreter.local_tts.indextts_server import chunk_text
        self.assertEqual(chunk_text("Short one, short two."), ["Short one, short two."])
        self.assertEqual(chunk_text("Thank you for the koha, it means a lot to our whānau.", 40),
                         ["Thank you for the koha,", "it means a lot to our whānau."])
        long = "a" * 60 + "."
        self.assertEqual(chunk_text(long), [long])          # no punctuation inside: kept whole, never mid-word


class TestKupuHints(unittest.TestCase):
    def test_every_concept_has_a_hint(self):
        self.assertEqual(set(PHONEMES), set(CONCEPTS))

    def test_annotate(self):
        self.assertEqual(annotate("My whānau are at the hui."),
                         f"My <whānau|{PHONEMES['ck:whanau']}> are at the <hui|{PHONEMES['ck:hui']}>.")
        self.assertEqual(annotate("Kia ora, WHANAU!"),
                         f"<Kia ora|{PHONEMES['ck:kia-ora']}>, <WHANAU|{PHONEMES['ck:whanau']}>!")
        self.assertEqual(annotate("My family are here."), "My family are here.")
        once = annotate("the hui and the hui")
        self.assertEqual(once.count("<hui|"), 2)
        self.assertEqual(annotate(once), once)              # already annotated text is left alone


def frame_with(text, listener="person:visitor_01"):
    text, spans = markers_to_spans(text)
    return InterpretationFrame(frame_id="x", direction="sign_to_speech", speaker_id="person:aroha",
                               addressee_id=listener, text=text, input_confidence=0.9,
                               emotion=Emotion("happy", 0.7, 0.4, 0.8), emotion_flow=spans)


class TestEngineAndRender(unittest.TestCase):
    def setUp(self):
        self.fake = FakeIndexTTS()
        self.addCleanup(self.fake.close)

    def test_health_and_synthesis(self):
        eng = IndexTTSEngine(self.fake.url, autostart=False)
        self.assertEqual((eng.native_sample_rate, eng.reference, eng.emotional), (22050, "ref.wav", True))
        pcm, rate = eng.synthesize_emotion("one two", VECTORS["sad"], 0.7)
        self.assertEqual((rate, len(pcm)), (22050, 2 * (2 * 2205 + 2 * 4410)))
        self.assertEqual(self.fake.requests[-1], {"text": "one two", "emo_vector": VECTORS["sad"], "emo_alpha": 0.7,
                                                  "lang": "EN"})

    def test_each_clause_gets_its_own_emotion(self):
        text, spans = markers_to_spans("{happy}Kia ora everyone, {sad}but my nan could not come.")
        frame = InterpretationFrame(frame_id="x", direction="sign_to_speech", speaker_id="person:aroha",
                                    addressee_id="person:visitor_01", text=text, input_confidence=0.9,
                                    emotion=Emotion("happy", 0.7, 0.4, 0.8), emotion_flow=spans)
        plan = plan_speech(frame, KGContext(demo_graph()))
        eng = IndexTTSEngine(self.fake.url, autostart=False)
        with tempfile.TemporaryDirectory() as d:
            res = render_speech(plan, eng, Path(d) / "x.wav")
            with wave.open(res.path) as w:
                self.assertEqual(w.getframerate(), 22050)
        self.assertEqual([r["text"] for r in self.fake.requests], ["Hello everyone,", "but my nan could not come."])
        self.assertEqual([(r["emo_vector"], r["emo_alpha"]) for r in self.fake.requests],
                         [(VECTORS["happy"], round(ALPHAS["happy"] * FLOW_STRENGTH, 3)),
                          (VECTORS["sad"], round(ALPHAS["sad"] * FLOW_STRENGTH, 3))])   # softened: emotion changes
        seg = res.params["segments"]
        self.assertEqual([(s["rate"], s["pitch"]) for s in seg], [(1.0, 1.0), (1.0, 1.0)])   # no pitch trick
        # each clause's 0.2 s padding is trimmed to 40 ms: 2 words = 0.2 s of sound + 0.08 s
        self.assertAlmostEqual(seg[0]["duration_s"], 0.2 + 0.08, delta=0.01)
        self.assertIn("IndexTTS emotion sad 0.80, melancholic 0.20 x strength 0.56 (softened", clause_line(2, seg[1]))

    def test_single_emotion_sentence_keeps_full_strength(self):
        plan = plan_speech(frame_with("{sad}I miss them. I really do."), KGContext(demo_graph()))
        with tempfile.TemporaryDirectory() as d:
            render_speech(plan, IndexTTSEngine(self.fake.url, autostart=False), Path(d) / "x.wav")
        self.assertEqual([r["emo_alpha"] for r in self.fake.requests], [ALPHAS["sad"], ALPHAS["sad"]])

    def test_kupu_hints_sent_only_when_te_reo_is_spoken(self):
        eng = IndexTTSEngine(self.fake.url, autostart=False)
        ctx = KGContext(demo_graph())
        with tempfile.TemporaryDirectory() as d:
            render_speech(plan_speech(frame_with("My whānau are here."), ctx), eng, Path(d) / "a.wav")
            res = render_speech(plan_speech(frame_with("My whānau are here.", "person:aroha"), ctx), eng, Path(d) / "b.wav")
            self.assertEqual(self.fake.requests[0]["text"], "My family are here.")          # English listener
            self.assertEqual(self.fake.requests[1]["text"], f"My <whānau|{PHONEMES['ck:whanau']}> are here.")
            self.assertIn("sent to IndexTTS: My <whānau|", clause_line(1, {**res.params, "duration_s": 1.0}))
            with mock.patch.object(config, "INDEXTTS_KUPU_HINTS", False):
                render_speech(plan_speech(frame_with("My whānau are here.", "person:aroha"), ctx), eng, Path(d) / "c.wav")
            self.assertEqual(self.fake.requests[2]["text"], "My whānau are here.")

    def test_plan_without_segments_uses_its_emotion(self):
        eng = IndexTTSEngine(self.fake.url, autostart=False)
        plan = plan_speech(InterpretationFrame(frame_id="x", direction="sign_to_speech", speaker_id="person:a",
                                               addressee_id="person:b", text="Where is it?", input_confidence=0.2,
                                               emotion=Emotion("confused", -0.2, 0.2, 0.6)), KGContext(demo_graph()))
        self.assertEqual(plan.mode, "clarify")
        with tempfile.TemporaryDirectory() as d:
            render_speech(plan, eng, Path(d) / "x.wav")
        self.assertEqual(self.fake.requests[-1]["emo_vector"], VECTORS["neutral"])

    def test_service_error_is_reported(self):
        bad = FakeIndexTTS(fail=True)
        self.addCleanup(bad.close)
        with self.assertRaisesRegex(RuntimeError, "boom"):
            IndexTTSEngine(bad.url, autostart=False).synthesize_emotion("hi", VECTORS["neutral"], 0.5)

    def test_unreachable_remote_service_is_not_autostarted(self):
        with self.assertRaisesRegex(RuntimeError, "not reachable"):
            IndexTTSEngine("http://192.0.2.1:9", autostart=True)     # TEST-NET address: never local


if __name__ == "__main__":
    unittest.main()
