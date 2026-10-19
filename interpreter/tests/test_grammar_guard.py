import contextlib
import io
import unittest

from interpreter.emotion import (GRAMMAR_DISCOUNT, LOOKALIKES, guard_grammar, nearest_label, prepare_emotion,
                                 question_type)
from interpreter.kg.demo import demo_graph
from interpreter.kg.provider import KGContext
from interpreter.run_pipeline import main as pipeline_main
from interpreter.task1 import plan_speech
from interpreter.testset import make_row


def frame(text, emotion, confidence=0.8):
    return make_row("x", text, emotion, confidence).frame


class TestQuestionType(unittest.TestCase):
    def test_types(self):
        self.assertEqual(question_type("Is the hui tomorrow?"), "yes/no")
        self.assertEqual(question_type("Which room is the hui in?"), "wh")
        self.assertEqual(question_type("Where is it?"), "wh")
        self.assertIsNone(question_type("The hui is tomorrow."))
        self.assertIsNone(question_type("Are you serious? Nobody told us!"))   # not a plain question
        self.assertEqual(set(LOOKALIKES), {"yes/no", "wh"})


class TestGuard(unittest.TestCase):
    def test_raised_brows_in_a_yes_no_question(self):
        f, notes = guard_grammar(frame("Is the hui tomorrow?", "surprised", 0.8))
        self.assertEqual((f.emotion.label, f.emotion.valence, f.emotion.arousal), ("neutral", 0.0, 0.0))
        self.assertAlmostEqual(f.emotion.confidence, 0.8 * GRAMMAR_DISCOUNT)
        self.assertIn("may be eyebrow grammar", notes[0])

    def test_furrowed_brows_in_a_wh_question(self):
        f, _ = guard_grammar(frame("Which room is the hui in?", "angry", 0.75))
        self.assertEqual(f.emotion.label, "neutral")
        f, _ = guard_grammar(frame("Which room is the hui in?", "confused", 0.75))
        self.assertEqual(f.emotion.label, "neutral")

    def test_untouched_cases(self):
        for text, emo in (("Where is my whānau?", "anxious"),          # not a look-alike
                          ("Is the hui tomorrow?", "angry"),           # angry is not a yes/no look-alike
                          ("The hui is tomorrow.", "surprised"),       # not a question
                          ("Which room is it?", "happy")):
            f = frame(text, emo)
            self.assertEqual(guard_grammar(f), (f, []), text)

    def test_only_the_question_span_is_guarded(self):
        f, notes = guard_grammar(frame("{surprised}Are you serious? {angry}Nobody told us!", "angry", 0.85))
        self.assertEqual(f.emotion.label, "angry")                      # whole sentence is not a question
        self.assertEqual([(s.start_word, s.emotion.label) for s in f.emotion_flow], [(0, "neutral"), (3, "angry")])
        self.assertEqual(len(notes), 1)

    def test_words_can_still_carry_the_emotion(self):
        f, _ = prepare_emotion(frame("Wow, really? You finished the whare?", "surprised", 0.85))
        self.assertEqual((f.emotion.label, f.emotion.confidence), ("surprised", 0.5))   # the "wow" text rule

    def test_guarded_question_sounds_neutral(self):
        f, _ = prepare_emotion(frame("Which room is the hui in?", "angry", 0.75))
        p = plan_speech(f, KGContext(demo_graph()))
        self.assertEqual(p.segments[0].emotion_label, "neutral")
        self.assertAlmostEqual(p.segments[0].prosody.rate, 1.0, places=2)

    def test_nearest_label(self):
        self.assertEqual(nearest_label(-0.7, 0.7), "angry")


class TestTrace(unittest.TestCase):
    def test_guard_shows_in_the_emotion_stage(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            pipeline_main(["--dry-run", "--testset", "--only", "g02,g04"])
        out = buf.getvalue()
        self.assertIn("grammar guard: wh question: the angry reading may be eyebrow grammar", out)
        self.assertIn("grammar guard: from word 0: yes/no question, the surprised reading", out)
        self.assertIn("neutral (guarded)", out)


if __name__ == "__main__":
    unittest.main()
