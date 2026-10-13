import csv
import io
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

from interpreter.context import StaticContext
from interpreter.contracts import EyeColour
from interpreter.mocks.mock_recogniser import load_frames
from interpreter.robot.items import Item, Variant, build_items
from interpreter.robot.listening_test import HEADER, run_session, shuffled
from interpreter.robot.pepper import REST, PepperOutput
from interpreter.robot.transport import DryRunTransport, QiTransport
from interpreter.task1 import plan_speech
from interpreter.tts.naoqi import render

FRAMES = Path(__file__).resolve().parent.parent / "mocks" / "frames"
PLANS = {f.frame_id: plan_speech(f, StaticContext()) for f in load_frames(FRAMES, "sign_to_speech")}


def names(t):
    return [(s, m) for s, m, _ in t.calls]


class TestAllowlist(unittest.TestCase):
    def test_disallowed_raises(self):
        t = DryRunTransport()
        for service, method in (("ALMotion", "moveTo"), ("ALTextToSpeech", "setParameter"),
                                ("ALRobotPosture", "goToPosture"), ("ALAutonomousLife", "setState")):
            with self.assertRaises(PermissionError):
                t.call(service, method)
        self.assertEqual(t.calls, [])

    def test_getters_return_canned_values(self):
        self.assertEqual(DryRunTransport().call("ALTextToSpeech", "getAvailableLanguages"), ["English"])


class TestPepperOutput(unittest.TestCase):
    def test_play_call_order(self):
        t = DryRunTransport()
        out = PepperOutput(t)
        out.start()
        rendered = out.play(PLANS["f-003"])   # solid eyes
        calls = t.calls
        self.assertEqual([c[:2] for c in calls[:2]], [("ALTextToSpeech", "setLanguage"), ("ALTextToSpeech", "setVolume")])
        say = next(i for i, c in enumerate(calls) if c[1] == "say")
        self.assertEqual(calls[say][2], (rendered.payload,))
        self.assertEqual(rendered.payload, render(PLANS["f-003"]).payload)
        self.assertEqual(calls[2][1], "fadeRGB")
        self.assertLess(2, say)
        last = calls[-1]
        self.assertEqual(last[1], "fadeRGB")
        r, g, b = (c / 255 for c in REST.rgb)
        self.assertEqual(last[2], ("FaceLeds", r, g, b, 0.5))
        self.assertNotIn("stopAll", [c[1] for c in calls])

    def test_pulse_runs_during_speech_and_stops(self):
        t = DryRunTransport(say_delay=0.6)
        before = threading.active_count()
        self.assertEqual(PLANS["f-001"].eye.pattern, "pulse")
        PepperOutput(t).play(PLANS["f-001"])
        say = next(i for i, c in enumerate(t.calls) if c[1] == "say")
        during = [c for c in t.calls[say:-1] if c[1] == "fadeRGB"]
        self.assertGreaterEqual(len(during), 2)
        self.assertEqual(threading.active_count(), before)
        n = len(t.calls)
        threading.Event().wait(0.5)
        self.assertEqual(len(t.calls), n)   # nothing fades after play returns

    def test_eyes_restored_when_say_raises(self):
        class Boom(DryRunTransport):
            def call(self, service, method, *args, post=False):
                if method == "say":
                    raise OSError("robot gone")
                return super().call(service, method, *args, post=post)
        t = Boom()
        with self.assertRaises(OSError):
            PepperOutput(t).play(PLANS["f-001"])
        self.assertEqual(t.calls[-1][1], "fadeRGB")
        self.assertEqual(t.calls[-1][2][4], 0.5)

    def test_fade_args_are_unit_floats_scaled_by_intensity(self):
        t = DryRunTransport()
        out = PepperOutput(t)
        out.set_eyes(EyeColour(rgb=(255, 128, 0), intensity=1.0, transition_ms=300))
        out.set_eyes(EyeColour(rgb=(255, 128, 0), intensity=0.5, transition_ms=300))
        full, half = (c[2] for c in t.calls)
        for args in (full, half):
            self.assertEqual(args[0], "FaceLeds")
            self.assertTrue(all(isinstance(x, float) and 0 <= x <= 1 for x in args[1:4]))
            self.assertEqual(args[4], 0.3)
        self.assertEqual(full[1], 1.0)
        for f, h in zip(full[1:4], half[1:4]):
            self.assertAlmostEqual(h, f / 2)

    def test_emergency_stop(self):
        t = DryRunTransport()
        PepperOutput(t).emergency_stop()
        self.assertEqual(names(t), [("ALTextToSpeech", "stopAll"), ("ALLeds", "fadeRGB")])


class TestItems(unittest.TestCase):
    def test_pack_shape(self):
        items = build_items()
        self.assertEqual([i.id for i in items], [
            "emotion-excited", "emotion-sad", "emotion-angry", "pronounce-whanau",
            "emphasis-whanau", "prosody-range", "clarify-cue"])
        for item in items:
            labels = [v.label for v in item.variants]
            self.assertEqual(len(labels), len(set(labels)), item.id)
            self.assertTrue(item.prompt)
            self.assertTrue(all(v.payload.strip() for v in item.variants), item.id)

    def test_shuffle_deterministic_and_rater_specific(self):
        items = build_items()
        order = lambda rater: [[v.label for v in shuffled(i, rater, 1)] for i in items]
        self.assertEqual(order("ana"), order("ana"))
        self.assertNotEqual(order("ana"), order("ben"))


class TestSession(unittest.TestCase):
    def test_csv_rows_skip_and_blank_rating(self):
        item = Item("x", "Title", "Prompt?", [Variant("one", "p1"), Variant("two", "p2")])
        first, second = (v.label for v in shuffled(item, "ana", 7))
        answers = iter([
            "s",                      # skip the first presented variant
            "", "",                   # play, then rate (no replay)
            "4", "", "5", "3",        # naturalness, emotion_match (blank), intelligibility, colour_fit
            "sounds fine",            # comment
        ])
        t, buf = DryRunTransport(), io.StringIO()
        w = csv.writer(buf)
        rows = run_session([item], PepperOutput(t), "ana", 7, lambda prompt: next(answers), w)
        self.assertEqual(rows, 1)
        (row,) = csv.reader(io.StringIO(buf.getvalue()))
        self.assertEqual(len(row), len(HEADER))
        self.assertEqual(row[1:], ["ana", "x", "B", second, "4", "", "5", "3", "sounds fine"])
        self.assertNotEqual(first, second)
        says = [c[2][0] for c in t.calls if c[1] == "say"]
        self.assertEqual(says, ["p1" if second == "one" else "p2"])
        self.assertEqual(names(t)[-2:], [("ALTextToSpeech", "stopAll"), ("ALLeds", "fadeRGB")])

    def test_stops_robot_on_interrupt(self):
        item = Item("x", "T", "P", [Variant("one", "p1")])
        def ask(prompt):
            raise KeyboardInterrupt
        t = DryRunTransport()
        self.assertEqual(run_session([item], PepperOutput(t), "ana", 0, ask, csv.writer(io.StringIO())), 0)
        self.assertIn(("ALTextToSpeech", "stopAll"), names(t))


class TestQiTransport(unittest.TestCase):
    def test_clear_error_without_qi(self):
        with mock.patch.dict(sys.modules, {"qi": None}):
            with self.assertRaises(RuntimeError) as cm:
                QiTransport("127.0.0.1")
        self.assertIn("qi", str(cm.exception))
        self.assertIn("2.7", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
