import base64
import contextlib
import io
import socketserver
import tempfile
import threading
import unittest
import wave
from array import array
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from interpreter import run_live
from interpreter.ai_server import client, stt
from interpreter.live import audio
from interpreter.local_tts import config, engines
from interpreter.tests.test_ai_server import AIServerCase, Orchestrator, stop

QUIET = array("h", [0] * 1600).tobytes()                      # 100 ms at 16 kHz
LOUD = array("h", [3000, -3000] * 800).tobytes()


def stt_server(transcript="my family is coming to the meeting", seen=None, error=None):
    """A fake speech-to-text instance speaking the orchestrator protocol."""
    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            req = client.receive_json(self.request)
            if seen is not None:
                seen.append(req)
            reply = {"robot_id": req["robot_id"], "request_id": req["request_id"]}
            if error or req["command"] != ["STT"]:
                reply.update(command=["ERR"], parameters={}, content=error or "Requested service is disabled on this instance")
            else:
                reply.update(command=[], parameters={}, content=f"  {transcript} ")
            client.send_json(self.request, reply)

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True
    server = Server(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class TestSpeechToText(unittest.TestCase):
    def test_transcribe_sends_the_pcm_and_strips_the_reply(self):
        seen = []
        server = stt_server(seen=seen)
        self.addCleanup(stop, server)
        text = stt.transcribe("127.0.0.1", server.server_address[1], LOUD * 3, 16000)
        self.assertEqual(text, "my family is coming to the meeting")
        req = seen[0]
        self.assertEqual((req["command"], req["parameters"]), (["STT"], dict(format="raw_pcm", dtype="int16", sample_rate=16000, channels=1)))
        self.assertEqual(base64.b64decode(req["content"]), LOUD * 3)

    def test_errors_say_what_to_check(self):
        with self.assertRaisesRegex(RuntimeError, "Could not reach the AI server for speech-to-text.*--stt-port"):
            stt.transcribe("127.0.0.1", 9, LOUD)
        server = stt_server(error="Requested service is disabled on this instance")
        self.addCleanup(stop, server)
        with self.assertRaisesRegex(RuntimeError, "speech-to-text failed.*disabled"):
            stt.transcribe("127.0.0.1", server.server_address[1], LOUD)

    def test_rms(self):
        self.assertEqual(stt.rms(b""), 0.0)
        self.assertAlmostEqual(stt.rms(LOUD), 3000)
        self.assertEqual(stt.rms(QUIET), 0)


class TestListening(unittest.TestCase):
    def test_speech_is_cut_out_with_a_little_before_and_the_silence_after(self):
        out = audio.utterance_from_chunks(iter([QUIET] * 20 + [LOUD] * 10 + [QUIET] * 15))
        self.assertEqual(len(out) // 3200, 3 + 10 + 10)         # 300 ms before, the speech, 1 s of trailing silence

    def test_nobody_spoke(self):
        self.assertIsNone(audio.utterance_from_chunks(iter([QUIET] * 200)))
        self.assertIsNone(audio.utterance_from_chunks(iter([])))

    def test_a_pause_inside_the_speech_does_not_end_it(self):
        out = audio.utterance_from_chunks(iter([LOUD] * 5 + [QUIET] * 5 + [LOUD] * 5 + [QUIET] * 12))
        self.assertEqual(len(out) // 3200, 5 + 5 + 5 + 10)

    def test_long_speech_is_cut_at_the_maximum(self):
        out = audio.utterance_from_chunks(iter([LOUD] * 500), max_ms=2000)
        self.assertLessEqual(len(out) // 3200, 21)

    def test_threshold_is_respected(self):
        self.assertIsNone(audio.utterance_from_chunks(iter([LOUD] * 50), threshold=5000, wait_ms=2000))


class TestMicrophoneLevel(unittest.TestCase):
    def test_threshold_follows_the_background_noise(self):
        self.assertEqual(audio.speech_threshold(1), 500)                      # a silent microphone: the minimum
        self.assertEqual(audio.speech_threshold(656), 2.5 * 656)              # the webcam's background
        self.assertEqual(audio.speech_threshold(100, minimum=300), 300)

    def test_measure_level_reads_a_second_of_audio(self):
        proc = mock.MagicMock()
        proc.stdout.read.return_value = LOUD * 10
        with mock.patch.object(audio, "_record_command", return_value=["parec"]), \
                mock.patch.object(audio.subprocess, "Popen", return_value=proc) as popen:
            self.assertAlmostEqual(audio.measure_level("mic", 1.0), 3000)
        proc.stdout.read.assert_called_once_with(32000)                      # 1 s of 16 kHz 16-bit mono
        proc.terminate.assert_called_once()
        self.assertTrue(popen.called)


class TestHelpers(unittest.TestCase):
    def test_auto_concepts_tags_english_words_that_have_a_kupu(self):
        self.assertEqual(run_live.auto_concepts("My family is coming to the meeting tomorrow."), "ck:whanau ck:hui")
        self.assertEqual(run_live.auto_concepts("Thanks for the food and the gift."), "ck:kai ck:koha")
        self.assertEqual(run_live.auto_concepts("Nothing here."), "")

    def test_make_frame_auto_emotion_is_weak_so_the_words_decide(self):
        f = run_live.make_frame(7, "I am so sorry.", "person:aroha", "person:aroha", None)
        self.assertEqual((f.frame_id, f.emotion.confidence, f.addressee_id), ("live-007", 0.0, "person:aroha"))
        g = run_live.make_frame(8, "I am so sorry.", "person:visitor_01", "person:aroha", "sad")
        self.assertEqual((g.emotion.label, g.emotion.confidence), ("sad", 0.8))


class FakeSink:
    def __init__(self, path):
        self.received, self.results, self.path = [], [], path

    def speak(self, plan):
        self.received.append(plan)
        self.results.append(SimpleNamespace(path=self.path, duration_s=1.5))


def args(**kw):
    base = dict(listener="person:visitor_01", signer="person:aroha", emotion=None, quiet=True, expressiveness=1.0,
                target_lang=None, carry=False, min_rms=500.0, stt_host="", stt_port=0, ai_host=None, push=False, mic=None)
    return SimpleNamespace(**{**base, **kw})


class TestSession(unittest.TestCase):
    def make(self, **kw):
        from interpreter.kg.demo import demo_graph
        self.played = []
        sink = FakeSink("/tmp/x.wav")
        return run_live.Session(args(**kw), demo_graph(), sink, self.played.append), sink

    def test_speak_runs_the_pipeline_and_plays_the_result(self):
        session, sink = self.make()
        with contextlib.redirect_stdout(io.StringIO()):
            session.speak("My family is coming to the meeting tomorrow.")
        self.assertEqual(sink.received[0].utterance, "My family is coming to the meeting tomorrow.")   # English listener
        self.assertEqual(self.played, ["/tmp/x.wav"])

    def test_a_listener_who_knows_te_reo_hears_the_kupu(self):
        session, sink = self.make(listener="person:aroha")
        with contextlib.redirect_stdout(io.StringIO()):
            session.speak("My family is coming to the meeting tomorrow.")
        self.assertEqual(sink.received[0].utterance, "My whānau is coming to the hui tomorrow.")

    def test_forced_emotion_and_commands(self):
        session, sink = self.make()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertTrue(session.command("/emotion angry"))
            session.speak("The results are in.")
            self.assertTrue(session.command("/listener person:aroha"))
            self.assertTrue(session.command("/listener person:nobody"))
            self.assertTrue(session.command("/emotion banana"))
            self.assertTrue(session.command("/emotion auto"))
            self.assertTrue(session.command("/again"))
            self.assertTrue(session.command("/help"))
            self.assertFalse(session.command("/quit"))
        self.assertEqual(sink.received[0].emotion_label, "angry")
        self.assertEqual((session.listener, session.emotion), ("person:aroha", None))
        text = out.getvalue()
        for want in ("emotion forced to angry", "listener is now person:aroha", "unknown person 'person:nobody'",
                     "unknown emotion", "emotion: automatic", "Enter = record"):
            self.assertIn(want, text)
        self.assertEqual(self.played, ["/tmp/x.wav", "/tmp/x.wav"])        # the turn, then /again replaying it


class TestWholeDemo(AIServerCase):
    """The real command line against a fake TTS and a fake STT server, with a fake microphone."""

    def setUp(self):
        super().setUp()
        self.orch = self.serve()
        self.stt = stt_server("my family is coming to the meeting tomorrow, but I am worried we are late")
        self.addCleanup(stop, self.stt)
        self.base = ["--ai-host", "127.0.0.1", "--ai-port", str(self.orch.port), "--stt-port", str(self.stt.server_address[1]),
                     "--no-play", "--quiet", "--out", tempfile.mkdtemp()]
        for name in ("AI_STT_HOST",):
            p = mock.patch.object(config, name, "")
            p.start()
            self.addCleanup(p.stop)

    def run_main(self, extra, inputs=()):
        buf = io.StringIO()
        with mock.patch("builtins.input", side_effect=[*inputs, EOFError()]), contextlib.redirect_stdout(buf):
            run_live.main(self.base + extra)
        return buf.getvalue()

    def test_start_up_calibrates_the_microphone_and_warns_about_a_silent_one(self):
        with mock.patch.object(audio, "measure_level", return_value=656.0):
            out = self.run_main([], inputs=["/quit"])
        self.assertIn("background level 656, so speech counts above 1640", out)
        self.assertNotIn("hears almost nothing", out)
        with mock.patch.object(audio, "measure_level", return_value=1.0):
            out = self.run_main([], inputs=["/level", "/quit"])
        self.assertIn("so speech counts above 500", out)
        self.assertIn("hears almost nothing", out)
        self.assertIn("level 1; speech counts above 500", out)

    def test_text_turn_skips_the_microphone(self):
        out = self.run_main(["--text", "Thanks so much, I am so happy."])
        self.assertIn("ready: speech-to-text 127.0.0.1:", out)
        self.assertIn("(emotion settings read)", out)
        self.assertIn('"Thanks so much, I am so happy."', out)
        texts = [c["text"] for c in self.orch.synth.calls]
        self.assertIn("Thanks so much, I am so happy.", texts)

    def test_wav_turn_goes_through_speech_to_text(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "in.wav"
            with wave.open(str(path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(LOUD * 10)
            out = self.run_main(["--wav", str(path), "--listener", "person:aroha"])
        self.assertIn('heard: "my family is coming to the meeting tomorrow, but I am worried we are late"', out)
        spoken = [c["text"] for c in self.orch.synth.calls[1:]]
        self.assertTrue(any("whānau" in t for t in spoken), spoken)           # Aroha knows te reo

    def test_interactive_loop_microphone_typing_and_quitting(self):
        with mock.patch.object(audio, "record_utterance", return_value=LOUD * 10):
            out = self.run_main([], inputs=["", "The results are in.", "/emotion sad", "/quit"])
        self.assertIn('heard: "my family is coming', out)                     # from the microphone + speech-to-text
        self.assertGreaterEqual(out.count("speech "), 2)                      # both turns were spoken
        self.assertIn("emotion forced to sad", out)

    def test_quiet_or_empty_recordings_are_not_sent(self):
        with mock.patch.object(audio, "record_utterance", side_effect=[None, QUIET * 10]):
            out = self.run_main([], inputs=["", "", "/quit"])
        self.assertIn("did not hear anything", out)
        self.assertIn("too quiet", out)
        self.assertNotIn("heard:", out)


if __name__ == "__main__":
    unittest.main()
