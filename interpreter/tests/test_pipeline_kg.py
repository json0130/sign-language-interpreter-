import copy
import unittest
from pathlib import Path

from interpreter.contracts import InterpretationFrame
from interpreter.emotion import CANONICAL_VA, MIN_CONFIDENCE, RULE_CONFIDENCE, RULES, guess_emotion, refine_emotion
from interpreter.kg.demo import demo_graph
from interpreter.kg.provider import KGContext
from interpreter.mocks.mock_recogniser import load_frames
from interpreter.pipeline import Pipeline
from interpreter.run_pipeline import build_pipeline
from interpreter.task1 import Task1Config, plan_speech
from interpreter.tests.test_contracts import frame_d1, frame_d2

KG_FRAMES = Path(__file__).resolve().parents[1] / "mocks" / "frames_kg"
CTX = KGContext(demo_graph())


def frame(text=None, listener="person:visitor_01", **emotion) -> InterpretationFrame:
    d = copy.deepcopy(frame_d1())
    d["addressee_id"] = listener
    if text:
        d["text"] = text
        del d["text_en"]
    d["emotion"].update(emotion)
    return InterpretationFrame.from_dict(d)


class Collect:
    def __init__(self):
        self.received = []

    def speak(self, plan):
        self.received.append(plan)

    send = speak


class TestEmotionBackup(unittest.TestCase):
    def test_confident_recogniser_is_kept(self):
        f = frame(confidence=MIN_CONFIDENCE)
        self.assertIs(refine_emotion(f), f)

    def test_low_confidence_uses_text_rules(self):
        f = frame("Thanks so much, that was really helpful!", label="neutral", valence=0, arousal=0, confidence=0.2)
        r = refine_emotion(f)
        self.assertEqual((r.emotion.label, r.emotion.confidence), ("happy", RULE_CONFIDENCE))
        self.assertEqual((r.emotion.valence, r.emotion.arousal), CANONICAL_VA["happy"])
        self.assertEqual(f.emotion.label, "neutral")   # the original frame is not mutated

    def test_low_confidence_without_a_rule_match_is_unchanged(self):
        f = frame("The results are in.", confidence=0.1)
        self.assertIs(refine_emotion(f), f)

    def test_each_rule_fires_on_an_example(self):
        examples = {"angry": "This is unacceptable.", "anxious": "I am worried we are running late.",
                    "sad": "I miss my family back home.", "excited": "We are over the moon!",
                    "happy": "Thanks for everything.", "surprised": "Wow, I did not expect that.",
                    "confused": "I don't understand the schedule.", "apologetic": "Sorry about that."}
        self.assertEqual({l for l, _ in RULES}, set(examples))
        for label, text in examples.items():
            self.assertEqual(guess_emotion(text), label, text)

    def test_deterministic(self):
        f = frame("Sorry, I am worried.", confidence=0.1)
        self.assertEqual(refine_emotion(f), refine_emotion(f))


class TestListenerWordChoice(unittest.TestCase):
    def test_unknown_listener_gets_english(self):
        p = plan_speech(frame(), CTX)
        self.assertEqual(p.utterance, "My family is coming to the meeting tomorrow.")
        self.assertEqual((p.prosody.emphasis, p.lexicon), (["family"], []))

    def test_known_listener_gets_te_reo(self):
        p = plan_speech(frame(listener="person:aroha"), CTX)
        self.assertEqual(p.utterance, "My whānau is coming to the hui tomorrow.")
        self.assertEqual(p.prosody.emphasis, ["whānau"])
        self.assertEqual({e.word for e in p.lexicon}, {"whānau", "hui"})

    def test_te_reo_in_the_transcript_is_glossed_for_an_unknown_listener(self):
        for text in ("My whānau is coming to the hui tomorrow.", "My whanau is coming to the hui tomorrow."):
            self.assertEqual(plan_speech(frame(text), CTX).utterance, "My family is coming to the meeting tomorrow.")

    def test_untagged_kupu_are_glossed_too(self):
        d = frame_d1(); d["text"] = "Kia ora, we are here for the hui."; del d["text_en"]; d["concepts"] = []
        d["addressee_id"] = "person:visitor_01"
        p = plan_speech(InterpretationFrame.from_dict(d), CTX)
        self.assertEqual(p.utterance, "Hello, we are here for the meeting.")

    def test_mixed_knowledge_is_per_kupu(self):
        # staff_01 has an explicit whānau edge but no hui knowledge
        p = plan_speech(frame(listener="person:staff_01"), CTX)
        self.assertEqual(p.utterance, "My whānau is coming to the meeting tomorrow.")

    def test_en_us_uses_the_en_us_gloss(self):
        p = plan_speech(frame("My whānau is coming."), CTX, target_lang="en-US")
        self.assertEqual((p.utterance, p.voice.language), ("My family is coming.", "en-US"))


class TestWholePipeline(unittest.TestCase):
    def run_frames(self, frames):
        speech, sign = Collect(), Collect()
        pipe = build_pipeline(demo_graph(), speech, sign, Task1Config())
        for f in frames:
            pipe.process(f)
        return speech.received, sign.received

    def test_te_reo_transcript_with_weak_emotion(self):
        (f,) = load_frames(KG_FRAMES)
        (plan,), _ = self.run_frames([f])
        self.assertEqual(plan.utterance, "Hello, my family are coming to the meeting tomorrow. Thanks so much for having us!")
        self.assertEqual(f.emotion.label, "neutral")           # what the recogniser said
        self.assertEqual(plan.emotion_label, "happy")          # what the text rules recovered
        self.assertGreater(plan.prosody.rate, 1)

    def test_mock_frames_route_both_directions(self):
        speech, sign = self.run_frames(load_frames(Path(__file__).resolve().parents[1] / "mocks" / "frames"))
        self.assertEqual((len(speech), len(sign)), (5, 5))

    def test_same_frame_twice_gives_the_same_plan(self):
        (f,) = load_frames(KG_FRAMES)
        (a, b), _ = self.run_frames([f, f])
        self.assertEqual(a, b)

    def test_text_to_speech_frames_are_spoken(self):
        d = frame_d2(); d["direction"] = "text_to_speech"
        (plan,), _ = self.run_frames([InterpretationFrame.from_dict(d)])
        self.assertEqual(plan.mode, "interpret")


if __name__ == "__main__":
    unittest.main()
