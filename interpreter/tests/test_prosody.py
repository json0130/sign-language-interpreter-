import argparse
import unittest

from interpreter.context import StaticContext
from interpreter.contracts import EMOTIONS, Emotion, InterpretationFrame
from interpreter.prosody_profiles import CLAMPS, PROFILES, strength
from interpreter.run_local import SWEEP_SENTENCE, SWEEP_VA, expressiveness_arg, sweep_frames
from interpreter.task1 import Task1Config, plan_speech

CTX = StaticContext()
DIMS = ("rate", "pitch", "volume")


def frame(label="happy", v=0.5, a=0.5, conf=0.9, text="We are here today."):
    return InterpretationFrame(frame_id="t", direction="sign_to_speech", speaker_id="person:a",
                               addressee_id="person:b", text=text, input_confidence=1.0,
                               emotion=Emotion(label, v, a, conf))


def prosody(label, v=0.5, a=0.5, conf=0.9, x=1.0):
    return plan_speech(frame(label, v, a, conf), CTX, Task1Config(expressiveness=x)).prosody


def sign(x):
    return (x > 1) - (x < 1)


class TestProfiles(unittest.TestCase):
    def test_every_emotion_has_a_profile_and_neutral_is_flat(self):
        self.assertEqual(set(PROFILES), set(EMOTIONS))
        self.assertEqual(PROFILES["neutral"], (1.0, 1.0, 1.0))

    def test_direction_matches_profile(self):
        for label, (v, a) in SWEEP_VA.items():
            p = prosody(label, v, a)
            for dim, m in zip(DIMS, PROFILES[label]):
                self.assertEqual(sign(getattr(p, dim)), sign(m), (label, dim))
        self.assertLess(prosody("sad", *SWEEP_VA["sad"]).rate, 1)
        self.assertGreater(prosody("excited", *SWEEP_VA["excited"]).pitch, 1)
        self.assertGreater(prosody("angry", *SWEEP_VA["angry"]).volume, 1)

    def test_emotions_are_clearly_different(self):
        rates = {l: prosody(l, *SWEEP_VA[l]).rate for l in SWEEP_VA}
        self.assertGreater(rates["excited"] - rates["sad"], 0.3)   # was ~0.12 with the old small tweaks

    def test_strength_formula(self):
        self.assertAlmostEqual(strength(1.0, 0.6, 0.8, 1.0), 1.0)
        self.assertAlmostEqual(strength(1.0, 0.0, 0.0, 0.0), 0.2)
        self.assertAlmostEqual(strength(2.0, 1.0, 1.0, 1.0), 2.0)   # intensity is capped at 1
        self.assertEqual(strength(0.0, 0.7, 0.7, 1.0), 0.0)


class TestBounds(unittest.TestCase):
    def test_never_leaves_clamps(self):
        vals = (-1, -0.5, 0, 0.5, 1)
        hit = False
        for label in EMOTIONS:
            for v in vals:
                for a in vals:
                    for conf in (0, 0.5, 1):
                        for x in (0, 1, 2):
                            p = prosody(label, v, a, conf, x)
                            for dim, (lo, hi) in CLAMPS.items():
                                val = getattr(p, dim)
                                self.assertTrue(lo <= val <= hi, (label, v, a, conf, x, dim, val))
                                hit = hit or val in (lo, hi)
        self.assertTrue(hit, "grid never reached a clamp, so this test would not catch a missing one")

    def test_expressiveness_zero_is_neutral_for_every_emotion(self):
        for label, (v, a) in SWEEP_VA.items():
            p = prosody(label, v, a, x=0.0)
            self.assertEqual((p.rate, p.pitch, p.volume), (1.0, 1.0, 1.0), label)

    def test_more_expressiveness_means_bigger_shift(self):
        for label in EMOTIONS:
            if label == "neutral":
                continue
            lo, hi = prosody(label, x=0.5), prosody(label, x=1.5)
            self.assertGreater(sum(abs(getattr(hi, d) - 1) for d in DIMS),
                               sum(abs(getattr(lo, d) - 1) for d in DIMS), label)


class TestConfidenceAndDeterminism(unittest.TestCase):
    def test_low_confidence_gives_weaker_shifts(self):
        for label, (v, a) in SWEEP_VA.items():
            if label == "neutral":
                continue
            hi, lo = prosody(label, v, a, conf=0.9), prosody(label, v, a, conf=0.2)
            for dim in DIMS:
                self.assertLess(abs(getattr(lo, dim) - 1), abs(getattr(hi, dim) - 1), (label, dim))

    def test_deterministic(self):
        f = frame("angry", -0.7, 0.7, text="I asked, and nobody came.")
        a, b = plan_speech(f, CTX), plan_speech(f, CTX)
        self.assertEqual(a, b)
        self.assertEqual(a.to_json(), b.to_json())

    def test_clarify_and_pauses_unchanged(self):
        f = frame("sad", -0.7, -0.4, text="I miss, you know, home.")
        self.assertEqual([x.ms for x in plan_speech(f, CTX).prosody.pauses], [300, 300])
        f.input_confidence = 0.2
        self.assertEqual(plan_speech(f, CTX).mode, "clarify")


class TestSweep(unittest.TestCase):
    def test_same_words_for_every_emotion(self):
        plans = [plan_speech(f, CTX) for f in sweep_frames(SWEEP_SENTENCE)]
        self.assertEqual([p.emotion_label for p in plans], list(SWEEP_VA))
        self.assertEqual(len({p.utterance for p in plans}), 1)
        self.assertEqual(plans[0].utterance, SWEEP_SENTENCE)
        self.assertEqual(len({(p.prosody.rate, p.prosody.pitch, p.prosody.volume) for p in plans}), len(plans))

    def test_sweep_uses_confidence_09(self):
        self.assertTrue(all(f.emotion.confidence == 0.9 for f in sweep_frames("Hi.")))

    def test_expressiveness_arg_range(self):
        self.assertEqual(expressiveness_arg("0"), 0.0)
        self.assertEqual(expressiveness_arg("2"), 2.0)
        for bad in ("-0.1", "2.01"):
            with self.assertRaises(argparse.ArgumentTypeError):
                expressiveness_arg(bad)


if __name__ == "__main__":
    unittest.main()
