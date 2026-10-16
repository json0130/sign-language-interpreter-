import contextlib
import io
import tempfile
import unittest
import wave
from array import array
from pathlib import Path

from interpreter.contracts import Emotion
from interpreter.emotion import CANONICAL_VA, RULE_CONFIDENCE
from interpreter.emotion_flow import (INERTIA, MARKER_CONFIDENCE, STATE_CARRY, Segment, SpeakerState, clauses,
                                      delivery, flow, join_wavs, nearest_label, plans, render, resample)
from interpreter.kg.demo import demo_graph
from interpreter.prosody_profiles import scaled, strength
from interpreter.run_pipeline import build_pipeline, main as pipeline_main
from interpreter.task1 import Task1Config
from interpreter.testset import DEFAULT_TESTSET, load_testset, make_row, strip_markers
from interpreter.tests.test_local_tts import FakeEngine

CSV = """id,mix,text,emotion,confidence,listener,concepts,notes
a1,english,My family is coming to the meeting tomorrow.,excited,0.85,person:visitor_01,ck:whanau ck:hui,
a2,english,My family is coming to the meeting tomorrow.,excited,0.85,person:aroha,ck:whanau ck:hui,
a3,english+kupu,My whānau are coming to the hui tomorrow.,excited,0.85,person:visitor_01,,
a4,english,"Thanks so much, that was really helpful!",neutral,0.2,person:visitor_01,,
a5,english+kupu,"Kia ora, thanks for the kai!",happy,0.8,person:visitor_01,,
a6,english+kupu,"{happy}Kia ora, {sad}but koro could not come.",happy,0.7,,,flow row
"""


def temp_csv(text: str) -> Path:
    d = tempfile.mkdtemp()
    p = Path(d) / "t.csv"
    p.write_text(text, encoding="utf-8")
    return p


class Collect:
    def __init__(self):
        self.received = []

    def speak(self, plan):
        self.received.append(plan)

    send = speak


class TestTestset(unittest.TestCase):
    def test_shipped_testset_is_valid(self):
        rows = load_testset(DEFAULT_TESTSET)
        self.assertTrue(10 <= len(rows) <= 20)
        self.assertEqual(len({r.frame.frame_id for r in rows}), len(rows))

    def test_rows_become_frames(self):
        rows = {r.frame.frame_id: r for r in load_testset(temp_csv(CSV))}
        a1 = rows["a1"].frame
        self.assertEqual((a1.addressee_id, a1.emotion.label, a1.emotion.confidence), ("person:visitor_01", "excited", 0.85))
        self.assertEqual((a1.emotion.valence, a1.emotion.arousal), CANONICAL_VA["excited"])
        self.assertEqual([(c.slug, c.english, c.te_reo) for c in a1.concepts],
                         [("ck:whanau", "family", "whānau"), ("ck:hui", "meeting", "hui")])
        self.assertEqual(rows["a6"].frame.text, "Kia ora, but koro could not come.")        # markers stripped
        self.assertEqual(rows["a6"].marked_text, "{happy}Kia ora, {sad}but koro could not come.")
        self.assertEqual(rows["a6"].frame.addressee_id, "person:visitor_01")             # blank -> default
        self.assertEqual(rows["a6"].notes, "flow row")

    def test_bad_rows_name_the_line(self):
        with self.assertRaisesRegex(ValueError, "line 2 \\(b1\\).*ck:nope"):
            load_testset(temp_csv("id,text,emotion,concepts\nb1,Hi.,happy,ck:nope\n"))
        with self.assertRaisesRegex(ValueError, "line 2 \\(b1\\)"):
            load_testset(temp_csv("id,text,emotion\nb1,Hi.,ecstatic\n"))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_testset(temp_csv("id,text,emotion\nb1,Hi.,happy\nb1,Hey.,happy\n"))

    def test_strip_markers(self):
        self.assertEqual(strip_markers("{calm}One. {sad}Two."), "One. Two.")

    def test_pipeline_output_per_row(self):
        speech = Collect()
        pipe = build_pipeline(demo_graph(), speech, Collect(), Task1Config())
        for r in load_testset(temp_csv(CSV)):
            pipe.process(r.frame)
        got = {p.frame_id: (p.utterance, p.emotion_label) for p in speech.received}
        self.assertEqual(got["a1"], ("My family is coming to the meeting tomorrow.", "excited"))
        self.assertEqual(got["a2"], ("My whānau is coming to the hui tomorrow.", "excited"))
        self.assertEqual(got["a3"], ("My family are coming to the meeting tomorrow.", "excited"))
        self.assertEqual(got["a4"], ("Thanks so much, that was really helpful!", "happy"))   # text rule
        self.assertEqual(got["a5"], ("Hello, thanks for the food!", "happy"))
        self.assertEqual(got["a6"][0], "Hello, but koro could not come.")

    def test_cli_testset_and_text(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            pipeline_main(["--dry-run", "--testset", str(temp_csv(CSV)), "--only", "a1,a4"])
            pipeline_main(["--dry-run", "--text", "My whānau loved the kai!", "--listener", "person:aroha",
                           "--emotion", "excited"])
        out = buf.getvalue()
        self.assertIn("SUMMARY", out)
        self.assertIn("happy (text rule)", out)
        self.assertIn('says: "My whānau loved the kai!"', out)


class TestFlow(unittest.TestCase):
    U = Emotion("calm", 0.4, -0.5, 0.6)

    def test_clauses(self):
        self.assertEqual(clauses("One thing. Then another, but not this. Done!"),
                         [(None, "One thing."), (None, "Then another,"), (None, "but not this."), (None, "Done!")])
        self.assertEqual(clauses("{happy}Kia ora, {sad}but koro could not come. Sadly."),
                         [("happy", "Kia ora,"), ("sad", "but koro could not come."), ("sad", "Sadly.")])
        self.assertEqual(clauses("Hello, nice to meet you."), [(None, "Hello, nice to meet you.")])
        with self.assertRaises(ValueError):
            clauses("{ecstatic}Hi.")

    def test_marker_is_used_as_written(self):
        (seg,) = flow("{angry}Stop.", self.U)
        self.assertEqual((seg.valence, seg.arousal, seg.confidence, seg.source), (-0.7, 0.7, MARKER_CONFIDENCE, "marker"))

    def test_rule_blends_with_the_utterance_emotion(self):
        (seg,) = flow("Thanks so much.", self.U)
        cu, cr = self.U.confidence, RULE_CONFIDENCE
        hv, ha = CANONICAL_VA["happy"]
        self.assertAlmostEqual(seg.valence, round((cu * 0.4 + cr * hv) / (cu + cr), 3))
        self.assertAlmostEqual(seg.arousal, round((cu * -0.5 + cr * ha) / (cu + cr), 3))
        self.assertEqual(seg.source, "rule")

    def test_utterance_fallback_and_inertia(self):
        a, b = flow("Thanks so much. The results are in.", self.U)   # a marker would also cover the second clause
        self.assertEqual((a.source, b.source), ("rule", "utterance"))
        self.assertAlmostEqual(b.valence, round((1 - INERTIA) * 0.4 + INERTIA * a.valence, 3))
        self.assertAlmostEqual(b.arousal, round((1 - INERTIA) * -0.5 + INERTIA * a.arousal, 3))

    def test_speaker_state_carry(self):
        (seg,) = flow("The results are in.", self.U, previous=(-0.7, 0.7))
        self.assertAlmostEqual(seg.valence, round((1 - STATE_CARRY) * 0.4 + STATE_CARRY * -0.7, 3))
        (marked,) = flow("{happy}The results are in.", self.U, previous=(-0.7, 0.7))
        self.assertEqual(marked.valence, CANONICAL_VA["happy"][0])   # an explicit marker is not carried
        state = SpeakerState()
        state.update("p", flow("{sad}Bye.", self.U))
        self.assertEqual(state.get("p"), CANONICAL_VA["sad"])
        self.assertIsNone(state.get("q"))

    def test_deterministic(self):
        text = "{happy}Kia ora, my whānau are here. I'm worried we're late."
        self.assertEqual(flow(text, self.U), flow(text, self.U))

    def test_delivery_matches_task1_at_canonical_points(self):
        for label, (v, a) in CANONICAL_VA.items():
            want = scaled(label, strength(1.0, v, a, 0.9))
            for got, exp in zip(delivery(v, a, 0.9), want):
                self.assertAlmostEqual(got, exp, places=3, msg=label)
            self.assertEqual(nearest_label(v, a), label)

    def test_delivery_between_two_emotions_is_in_between(self):
        sad, exc = CANONICAL_VA["sad"], CANONICAL_VA["excited"]
        mid = ((sad[0] + exc[0]) / 2, (sad[1] + exc[1]) / 2)
        r_sad, r_mid, r_exc = (delivery(*p, 1.0)[0] for p in (sad, mid, exc))
        self.assertLess(r_sad, r_mid)
        self.assertLess(r_mid, r_exc)

    def test_plans_one_per_clause(self):
        segs = flow("{happy}Kia ora, my whānau are here, {sad}but koro could not come.", self.U)
        ps = plans("x", segs)
        self.assertEqual([p.frame_id for p in ps], ["x-1", "x-2"])
        self.assertEqual({e.word for e in ps[0].lexicon}, {"kia ora", "whānau"})
        self.assertGreater(ps[0].prosody.rate, ps[1].prosody.rate)

    def test_resample(self):
        flat = array("h", [1000] * 1600)
        out = resample(flat, 16000, 24000)
        self.assertEqual(len(out), 2400)
        self.assertTrue(all(x == 1000 for x in out))
        self.assertEqual(resample(flat, 16000, 16000), flat)

    def test_render_joins_clauses_at_one_rate(self):
        segs = flow("{sad}One two three. {excited}Four five six.", self.U)
        with tempfile.TemporaryDirectory() as d:
            out = render("x", segs, Path(d), "en-NZ", 1.0, engine=FakeEngine())
            parts = sorted((Path(d) / "parts").glob("*.wav"))
            durations, rates = [], []
            for p in parts:
                with wave.open(str(p)) as w:
                    durations.append(w.getnframes() / w.getframerate())
                    rates.append(w.getframerate())
            with wave.open(str(out)) as w:
                total, rate = w.getnframes() / w.getframerate(), w.getframerate()
        self.assertEqual(len(parts), 2)
        self.assertNotEqual(rates[0], rates[1])            # pitch differs per clause
        self.assertEqual(rate, max(rates))
        self.assertAlmostEqual(total, sum(durations) + 0.12, delta=0.01)

    def test_join_wavs_single_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.wav"
            with wave.open(str(p), "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000)
                w.writeframes(array("h", [5] * 8000).tobytes())
            self.assertAlmostEqual(join_wavs([p], Path(d) / "j.wav"), 1.0)


if __name__ == "__main__":
    unittest.main()
