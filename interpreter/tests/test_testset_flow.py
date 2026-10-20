import contextlib
import csv
import io
import json
import tempfile
import unittest
import wave
from array import array
from dataclasses import replace
from pathlib import Path
from unittest import mock

from interpreter.context import StaticContext
from interpreter.contracts import (Emotion, EmotionSpan, InterpretationFrame, Prosody, SpeechPlan, SpeechSegment)
from interpreter.emotion import CANONICAL_VA, RULE_CONFIDENCE
from interpreter.emotion_flow import (INERTIA, MARKER_CONFIDENCE, STATE_CARRY, SpeakerState, clauses, delivery, flow,
                                      markers_to_spans, nearest_label)
from interpreter.kg.demo import demo_graph
from interpreter.kg.provider import KGContext
from interpreter.local_tts.render import join_wavs, render_speech, resample
from interpreter.prosody_profiles import scaled, strength
from interpreter.run_pipeline import build_pipeline, main as pipeline_main
from interpreter.task1 import Task1Config, plan_speech
from interpreter.testset import DEFAULT_TESTSET, load_testset, strip_markers
from interpreter.tests.test_local_tts import FakeEngine

CSV = """id,mix,text,emotion,confidence,listener,concepts,notes
a1,english,My family is coming to the meeting tomorrow.,excited,0.85,person:visitor_01,ck:whanau ck:hui,
a2,english,My family is coming to the meeting tomorrow.,excited,0.85,person:aroha,ck:whanau ck:hui,
a3,english+kupu,My whānau are coming to the hui tomorrow.,excited,0.85,person:visitor_01,,
a4,english,"Thanks so much, that was really helpful!",neutral,0.2,person:visitor_01,,
a5,english+kupu,"Kia ora, thanks for the kai!",happy,0.8,person:visitor_01,,
a6,english+kupu,"{happy}Kia ora, {sad}but my nan could not come.",happy,0.7,,,flow row
"""
U = Emotion("calm", 0.4, -0.5, 0.6)


def temp_csv(text: str) -> Path:
    p = Path(tempfile.mkdtemp()) / "t.csv"
    p.write_text(text, encoding="utf-8")
    return p


def frame(text: str, spans=(), listener="person:visitor_01", emotion=U) -> InterpretationFrame:
    return InterpretationFrame(frame_id="x", direction="sign_to_speech", speaker_id="person:aroha",
                               addressee_id=listener, text=text, emotion=emotion, input_confidence=0.9,
                               emotion_flow=list(spans))


class Collect:
    def __init__(self):
        self.received = []

    def speak(self, plan):
        self.received.append(plan)

    send = speak


class TestTestset(unittest.TestCase):
    def test_shipped_testset_is_valid_and_uses_known_kupu_only(self):
        rows = load_testset(DEFAULT_TESTSET)
        self.assertGreaterEqual(len(rows), 20)
        self.assertEqual(len({r.frame.frame_id for r in rows}), len(rows))
        ctx = StaticContext(default=False)   # nobody knows te reo: every kupu must be replaceable by the KG
        for r in rows:
            spoken = plan_speech(r.frame, ctx).utterance
            for kupu in ("whānau", "hui", "kia ora", "kai", "mahi", "tamariki", "whare", "kōrero", "koha", "koro"):
                self.assertNotIn(kupu, spoken.lower(), r.frame.frame_id)

    def test_rows_become_frames(self):
        rows = {r.frame.frame_id: r for r in load_testset(temp_csv(CSV))}
        a1 = rows["a1"].frame
        self.assertEqual((a1.addressee_id, a1.emotion.label, a1.emotion.confidence), ("person:visitor_01", "excited", 0.85))
        self.assertEqual((a1.emotion.valence, a1.emotion.arousal), CANONICAL_VA["excited"])
        self.assertEqual([(c.slug, c.english, c.te_reo) for c in a1.concepts],
                         [("ck:whanau", "family", "whānau"), ("ck:hui", "meeting", "hui")])
        a6 = rows["a6"]
        self.assertEqual(a6.frame.text, "Kia ora, but my nan could not come.")
        self.assertEqual([(s.start_word, s.emotion.label) for s in a6.frame.emotion_flow], [(0, "happy"), (2, "sad")])
        self.assertEqual(a6.marked_text, "{happy}Kia ora, {sad}but my nan could not come.")
        self.assertEqual((a6.frame.addressee_id, a6.notes), ("person:visitor_01", "flow row"))

    def test_bad_rows_name_the_line(self):
        with self.assertRaisesRegex(ValueError, "line 2 \\(b1\\).*ck:nope"):
            load_testset(temp_csv("id,text,emotion,concepts\nb1,Hi.,happy,ck:nope\n"))
        with self.assertRaisesRegex(ValueError, "line 2 \\(b1\\)"):
            load_testset(temp_csv("id,text,emotion\nb1,Hi.,ecstatic\n"))
        with self.assertRaisesRegex(ValueError, "line 2 \\(b1\\).*marker"):
            load_testset(temp_csv("id,text,emotion\nb1,{ecstatic}Hi.,happy\n"))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_testset(temp_csv("id,text,emotion\nb1,Hi.,happy\nb1,Hey.,happy\n"))

    def test_pipeline_output_per_row(self):
        speech = Collect()
        pipe = build_pipeline(demo_graph(), speech, Collect(), Task1Config())
        for r in load_testset(temp_csv(CSV)):
            pipe.process(r.frame)
        got = {p.frame_id: p for p in speech.received}
        self.assertEqual(got["a1"].utterance, "My family is coming to the meeting tomorrow.")
        self.assertEqual(got["a2"].utterance, "My whānau is coming to the hui tomorrow.")
        self.assertEqual(got["a3"].utterance, "My family are coming to the meeting tomorrow.")
        self.assertEqual((got["a4"].utterance, got["a4"].emotion_label), ("Thanks so much, that was really helpful!", "happy"))
        self.assertEqual(got["a5"].utterance, "Hello, thanks for the food!")
        self.assertEqual([(s.text, s.emotion_label, s.source) for s in got["a6"].segments],
                         [("Hello,", "happy", "recogniser"), ("but my nan could not come.", "sad", "recogniser")])

    def test_cli_testset_and_text(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            pipeline_main(["--dry-run", "--testset", str(temp_csv(CSV)), "--only", "a4,a6"])
            pipeline_main(["--dry-run", "--text", "My whānau loved the kai!", "--listener", "person:aroha",
                           "--emotion", "excited"])
        out = buf.getvalue()
        for want in ("SUMMARY", "happy (text rule)", "recogniser emotion changes at word 0: happy, word 2: sad",
                     "emotion flow (2 clauses)", 'says: "My whānau loved the kai!"'):
            self.assertIn(want, out)


class TestMarkersAndClauses(unittest.TestCase):
    def test_markers_to_spans(self):
        text, spans = markers_to_spans("{happy}Kia ora everyone, {excited}my whānau are here, {sad}but nan is not.")
        self.assertEqual(text, "Kia ora everyone, my whānau are here, but nan is not.")
        self.assertEqual([(s.start_word, s.emotion.label, s.emotion.confidence) for s in spans],
                         [(0, "happy", MARKER_CONFIDENCE), (3, "excited", MARKER_CONFIDENCE), (7, "sad", MARKER_CONFIDENCE)])
        self.assertEqual(strip_markers("{calm}One. {sad} Two."), "One. Two.")
        with self.assertRaises(ValueError):
            markers_to_spans("{ecstatic}Hi.")

    def test_clauses(self):
        self.assertEqual([(c.start, c.text) for c in clauses("One thing. Then another, but not this. Done!")],
                         [(0, "One thing."), (2, "Then another,"), (4, "but not this."), (7, "Done!")])
        self.assertEqual([c.text for c in clauses("Hello, nice to meet you.")], ["Hello, nice to meet you."])
        self.assertEqual([c.text for c in clauses("Hi there, and then we left.")], ["Hi there,", "and then we left."])
        self.assertEqual([c.text for c in clauses("I came here today", breaks=[2])], ["I came", "here today"])


class TestFlow(unittest.TestCase):
    def test_recogniser_span_is_used_as_given(self):
        span = EmotionSpan(0, Emotion("angry", -0.6, 0.5, 0.8))
        (seg,) = flow("Stop.", U, [span])
        self.assertEqual((seg.valence, seg.arousal, seg.confidence, seg.source), (-0.6, 0.5, 0.8, "recogniser"))

    def test_span_covers_following_clauses_until_the_next(self):
        _, spans = markers_to_spans("{angry}Stop that. Right now. {calm}Thank you.")
        segs = flow("Stop that. Right now. Thank you.", U, spans)
        self.assertEqual([s.label for s in segs], ["angry", "angry", "calm"])

    def test_rule_blends_with_the_sentence_emotion(self):
        (seg,) = flow("Thanks so much.", U)
        cu, cr = U.confidence, RULE_CONFIDENCE
        hv, ha = CANONICAL_VA["happy"]
        self.assertAlmostEqual(seg.valence, round((cu * U.valence + cr * hv) / (cu + cr), 3))
        self.assertAlmostEqual(seg.arousal, round((cu * U.arousal + cr * ha) / (cu + cr), 3))
        self.assertEqual(seg.source, "rule")

    def test_rule_with_zero_confidence_sentence(self):
        (seg,) = flow("Thanks so much.", Emotion("neutral", 0, 0, 0.0))
        self.assertEqual((seg.valence, seg.arousal), CANONICAL_VA["happy"])

    def test_sentence_fallback_and_inertia(self):
        a, b = flow("Thanks so much. The results are in.", U)
        self.assertEqual((a.source, b.source), ("rule", "utterance"))
        self.assertAlmostEqual(b.valence, round((1 - INERTIA) * U.valence + INERTIA * a.valence, 3))

    def test_speaker_state_carry(self):
        (seg,) = flow("The results are in.", U, previous=(-0.7, 0.7))
        self.assertAlmostEqual(seg.valence, round((1 - STATE_CARRY) * U.valence + STATE_CARRY * -0.7, 3))
        _, spans = markers_to_spans("{happy}The results are in.")
        (marked,) = flow("The results are in.", U, spans, previous=(-0.7, 0.7))
        self.assertEqual(marked.valence, CANONICAL_VA["happy"][0])   # a recogniser span is not carried
        state = SpeakerState()
        state.update("p", 0.1, 0.2)
        self.assertEqual((state.get("p"), state.get("q")), ((0.1, 0.2), None))

    def test_deterministic(self):
        text = "Kia ora, my whānau are here. I'm worried we're late."
        self.assertEqual(flow(text, U), flow(text, U))

    def test_delivery_matches_the_profiles_at_canonical_points(self):
        for label, (v, a) in CANONICAL_VA.items():
            for got, exp in zip(delivery(v, a, 0.9), scaled(label, strength(1.0, v, a, 0.9))):
                self.assertAlmostEqual(got, exp, places=3, msg=label)
            self.assertEqual(nearest_label(v, a), label)

    def test_delivery_between_two_emotions_is_in_between(self):
        sad, exc = CANONICAL_VA["sad"], CANONICAL_VA["excited"]
        mid = ((sad[0] + exc[0]) / 2, (sad[1] + exc[1]) / 2)
        r_sad, r_mid, r_exc = (delivery(*p, 1.0)[0] for p in (sad, mid, exc))
        self.assertLess(r_sad, r_mid)
        self.assertLess(r_mid, r_exc)


class TestTask1Segments(unittest.TestCase):
    CTX = KGContext(demo_graph())

    def test_segments_cover_the_utterance_and_use_the_listeners_words(self):
        text, spans = markers_to_spans("{happy}Kia ora, my whānau are here, {sad}but my nan could not come.")
        p = plan_speech(frame(text, spans), self.CTX)
        self.assertEqual(p.utterance, "Hello, my family are here, but my nan could not come.")
        self.assertEqual([(s.text, s.emotion_label) for s in p.segments],
                         [("Hello, my family are here,", "happy"), ("but my nan could not come.", "sad")])
        self.assertGreater(p.segments[0].prosody.rate, p.segments[1].prosody.rate)
        self.assertEqual([x.after_word_index for x in p.segments[0].prosody.pauses], [0])   # comma inside the clause
        self.assertEqual(p.lexicon, [])

    def test_known_listener_keeps_te_reo_in_segments(self):
        p = plan_speech(frame("My whānau are here. Thanks so much!", listener="person:aroha"), self.CTX)
        self.assertEqual([s.text for s in p.segments], ["My whānau are here.", "Thanks so much!"])

    def test_previous_state_moves_the_first_clause(self):
        a = plan_speech(frame("The results are in."), self.CTX)
        b = plan_speech(frame("The results are in."), self.CTX, previous=(-0.7, 0.7))
        self.assertNotEqual(a.segments[0].valence, b.segments[0].valence)

    def test_clarify_has_no_segments(self):
        f = replace(frame("Where is it?"), input_confidence=0.2)
        self.assertEqual(plan_speech(f, self.CTX).segments, [])


class TestContract(unittest.TestCase):
    def test_frame_emotion_flow_roundtrip_and_validation(self):
        _, spans = markers_to_spans("{happy}Hi there, {sad}but bye.")
        f = frame("Hi there, but bye.", spans)
        self.assertEqual(InterpretationFrame.from_dict(json.loads(f.to_json())), f)
        with self.assertRaises(ValueError):
            frame("Hi.", [EmotionSpan(5, U)])                        # beyond the text
        with self.assertRaises(ValueError):
            frame("Hi there friend.", [EmotionSpan(2, U), EmotionSpan(1, U)])   # not increasing
        with self.assertRaises(ValueError):
            EmotionSpan(-1, U)

    def test_plan_segments_roundtrip_and_validation(self):
        p = plan_speech(frame("Thanks so much. The results are in."), KGContext(demo_graph()))
        self.assertEqual(SpeechPlan.from_dict(json.loads(p.to_json())), p)
        with self.assertRaises(ValueError):
            replace(p, segments=p.segments[:1])                       # does not cover the utterance
        with self.assertRaises(ValueError):
            SpeechSegment("Hi.", "happy", source="guess")
        with self.assertRaises(ValueError):
            SpeechSegment("Hi.", "happy", Prosody(emphasis=["bye"]))


class TestRendering(unittest.TestCase):
    def test_resample(self):
        flat = array("h", [1000] * 1600)
        out = resample(flat, 16000, 24000)
        self.assertEqual(len(out), 2400)
        self.assertTrue(all(x == 1000 for x in out))
        self.assertEqual(resample(flat, 16000, 16000), flat)

    def test_segments_render_one_clause_each_and_join(self):
        text, spans = markers_to_spans("{sad}One two three. {excited}Four five six.")
        p = plan_speech(frame(text, spans), StaticContext())
        eng = FakeEngine()
        with tempfile.TemporaryDirectory() as d:
            res = render_speech(p, eng, Path(d) / "x.wav")
            with wave.open(res.path) as w:
                total, rate = w.getnframes() / w.getframerate(), w.getframerate()
        self.assertEqual([t for t, _ in eng.calls], ["One two three.", "Four five six."])
        seg = res.params["segments"]
        self.assertEqual(len(seg), 2)
        self.assertNotEqual(seg[0]["output_sample_rate"], seg[1]["output_sample_rate"])   # pitch differs per clause
        self.assertEqual(rate, max(x["output_sample_rate"] for x in seg))
        self.assertAlmostEqual(total, seg[0]["duration_s"] + seg[1]["duration_s"] + 0.30, delta=0.01)   # "." ends clause 1: 300 ms

    def test_plan_without_segments_renders_as_one(self):
        p = replace(plan_speech(frame("Thanks so much. The results are in."), StaticContext()), segments=[])
        eng = FakeEngine()
        with tempfile.TemporaryDirectory() as d:
            res = render_speech(p, eng, Path(d) / "x.wav")
        self.assertEqual(len(eng.calls), 1)
        self.assertNotIn("segments", res.params)

    def test_pause_length_depends_on_how_the_clause_ends(self):
        from interpreter.local_tts.render import gap_after
        self.assertEqual([gap_after(t) for t in ("Which room?", "Done.", "Wow!", "Well,", "and so", "Done. ")],
                         [450, 300, 300, 120, 120, 300])

    def test_join_wavs_accepts_one_gap_per_join(self):
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for i in range(3):
                p = Path(d) / f"{i}.wav"
                with wave.open(str(p), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(8000)
                    w.writeframes(array("h", [5] * 8000).tobytes())
                paths.append(p)
            self.assertAlmostEqual(join_wavs(paths, Path(d) / "a.wav", [100, 500]), 3.6)
            self.assertAlmostEqual(join_wavs(paths, Path(d) / "b.wav", 200), 3.4)

    def test_join_wavs_single_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.wav"
            with wave.open(str(p), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(8000)
                w.writeframes(array("h", [5] * 8000).tobytes())
            self.assertAlmostEqual(join_wavs([p], Path(d) / "j.wav"), 1.0)


def run_interactive(answers, *argv):
    """Run the CLI with scripted keyboard input (then end-of-input, like Ctrl-D); returns (stdout plus every prompt
    shown, frame ids in the order they were run)."""
    buf = io.StringIO()
    with mock.patch("builtins.input", side_effect=[*answers, EOFError()]) as fake, contextlib.redirect_stdout(buf):
        pipeline_main(["--dry-run", "--testset", str(temp_csv(CSV)), *argv])
    out = buf.getvalue() + "\n".join(c.args[0] for c in fake.call_args_list)
    return out, [line.split()[1] for line in buf.getvalue().splitlines() if line.startswith("===== ")]


class TestStepAndScore(unittest.TestCase):
    def test_step_next_previous_replay_jump_quit(self):
        out, visited = run_interactive(["", "p", "r", "a5", "q"], "--step", "--only", "a1,a2,a5")
        self.assertEqual(visited, ["a1", "a2", "a1", "a5"])
        self.assertIn("[1/3] Enter/n = next", out)

    def test_previous_on_first_and_unknown_command(self):
        out, visited = run_interactive(["p", "zzz", "n", "n"], "--step", "--only", "a1,a2")
        self.assertEqual(visited, ["a1", "a2"])                 # Enter/n after the last one ends the run
        self.assertIn("this is the first one", out)
        self.assertIn("not understood: 'zzz'", out)

    def test_end_of_input_stops_cleanly(self):
        out, visited = run_interactive([], "--step", "--only", "a1,a2")
        self.assertEqual(visited, ["a1"])
        self.assertIn("stopped", out)

    def test_score_single_and_flow_sentences(self):
        path = Path(tempfile.mkdtemp()) / "scores.csv"
        single = ["y", "y", "y", "n", "y", "too slow"]            # words, maori, emotion, natural, not overdone, comment
        flow_answers = ["y", "r", "y", "y", "y", "s", "y", ""]     # words, (replay) maori, flow, natural, joins skipped, ...
        out, visited = run_interactive(single + [""] + flow_answers + ["q"],
                                       "--score", str(path), "--only", "a4,a6")
        self.assertEqual(visited, ["a4", "a6"])
        rows = {r["id"]: r for r in csv.DictReader(open(path, encoding="utf-8"))}
        self.assertEqual((rows["a4"]["emotion"], rows["a4"]["flow"], rows["a4"]["joins"]), ("yes", "n/a", "n/a"))
        self.assertEqual((rows["a4"]["natural"], rows["a4"]["verdict"], rows["a4"]["comment"]), ("no", "no", "too slow"))
        self.assertEqual((rows["a6"]["emotion"], rows["a6"]["flow"], rows["a6"]["joins"]), ("n/a", "yes", "skipped"))
        self.assertEqual(rows["a6"]["verdict"], "incomplete")
        self.assertIn("SCORES (dry-run): 0/2 sentences passed every check", out)
        self.assertIn("failed: a4", out)

    def test_rescoring_a_sentence_replaces_its_answers(self):
        path = Path(tempfile.mkdtemp()) / "scores.csv"
        no = ["n", "y", "y", "y", "y", ""]
        yes = ["y", "y", "y", "y", "y", ""]
        run_interactive(no + ["a1"] + yes + ["q"], "--score", str(path), "--only", "a1")
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        self.assertEqual([(r["id"], r["verdict"]) for r in rows], [("a1", "yes")])


if __name__ == "__main__":
    unittest.main()
