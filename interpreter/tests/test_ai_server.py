import base64
import contextlib
import importlib.util
import io
import math
import socketserver
import tempfile
import threading
import unittest
import wave
from array import array
from pathlib import Path
from unittest import mock

from interpreter.ai_server import check, client
from interpreter.ai_server.tts_engine import NEUTRAL, AIServerTTSEngine
from interpreter.contracts import Emotion, InterpretationFrame
from interpreter.emotion_flow import markers_to_spans
from interpreter.kg.demo import demo_graph
from interpreter.kg.provider import KGContext
from interpreter.local_tts import config, engines
from interpreter.emotion import CANONICAL_VA
from interpreter.local_tts.indextts_emotion import ALPHAS, VECTORS, transition_scale
from interpreter.local_tts.indextts_phonemes import PHONEMES
from interpreter.local_tts.indextts_server import orchestrator_reply, serve_orchestrator, wav_bytes
from interpreter.local_tts.render import render_speech
from interpreter.run_pipeline import main as pipeline_main
from interpreter.task1 import plan_speech
from interpreter.tests.test_testset_flow import CSV, temp_csv

LIBRARY_TTS = Path(__file__).resolve().parents[3] / "Library-Book-Robot" / "vendor" / "ai_server" / "tts.py"


class FakeSynth:
    """Stands in for IndexTTS: a tone of 0.1 s per word. Records every call."""

    def __init__(self):
        self.calls = []

    def __call__(self, text, vector, alpha, lang="EN"):
        self.calls.append({"text": text, "vector": vector, "alpha": alpha, "lang": lang})
        n = 2205 * max(1, len(text.split()))
        return 22050, array("h", (round(8000 * math.sin(i / 10)) for i in range(n))).tobytes()


class Orchestrator:
    """Our real protocol server (indextts_server.serve_orchestrator) around a fake synth."""

    def __init__(self, synth=None):
        self.synth = synth or FakeSynth()
        self.server = serve_orchestrator("127.0.0.1", 0, self.synth)
        self.port = self.server.server_address[1]

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def other_engine_server(engine_name=None, echo=False):
    """A server that ignores the emotion fields: answers every TTS request, valid or not, with a tone. It names another
    engine, or (echo=True) echoes the request parameters back the way the AI server's orchestrator did on 2026-10-20."""
    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            req = client.receive_json(self.request)
            params = {"format": "wav", "sample_rate": 24000, "channels": 1, "dtype": "int16"}
            if engine_name:
                params["engine"] = engine_name
            if echo:
                params.update(req["parameters"])
            tone = array("h", (round(6000 * math.sin(i / 10)) for i in range(4800))).tobytes()
            client.send_json(self.request, {"robot_id": req["robot_id"], "request_id": req["request_id"], "command": [],
                                            "parameters": params,
                                            "content": base64.b64encode(wav_bytes(24000, tone)).decode()})

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True
    server = Server(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def stop(server):
    server.shutdown()
    server.server_close()


def frame_for(text, listener="person:visitor_01"):
    text, spans = markers_to_spans(text)
    return InterpretationFrame(frame_id="x", direction="sign_to_speech", speaker_id="person:aroha", addressee_id=listener,
                               text=text, input_confidence=0.9, emotion=Emotion("happy", 0.7, 0.4, 0.8), emotion_flow=spans)


class AIServerCase(unittest.TestCase):
    def setUp(self):
        self.addCleanup(engines._ENGINES.clear)
        for name in ("AI_SERVER_HOST", "AI_SERVER_PORT"):
            p = mock.patch.object(config, name, getattr(config, name))
            p.start()
            self.addCleanup(p.stop)

    def serve(self, **kw):
        orch = Orchestrator(**kw)
        self.addCleanup(orch.close)
        config.AI_SERVER_HOST, config.AI_SERVER_PORT = "127.0.0.1", orch.port
        return orch


class TestProtocol(AIServerCase):
    def test_tts_roundtrip_with_the_client(self):
        orch = self.serve()
        vec = VECTORS["sad"]
        resp = client.request("127.0.0.1", orch.port, "TTS", {"emo_vector": vec, "emo_alpha": 0.7, "lang": "EN"}, "one two")
        pcm, rate = client.decode_wav(resp["content"], resp["parameters"])
        self.assertEqual((rate, len(pcm)), (22050, 2 * 2 * 2205))
        self.assertEqual(resp["parameters"]["engine"], "indextts")
        self.assertEqual(orch.synth.calls, [{"text": "one two", "vector": vec, "alpha": 0.7, "lang": "EN"}])

    def test_errors_come_back_as_err_responses(self):
        orch = self.serve()
        for command, params, text, expect in (("LLM", {}, "hi", "only does TTS"),
                                              ("TTS", {"emo_vector": [0.5] * 3}, "hi", "8 values"),
                                              ("TTS", {"emo_alpha": 2}, "hi", "between 0 and 1"),
                                              ("TTS", {}, "x" * 2001, "1 to 2000")):
            with self.assertRaisesRegex(ValueError, f"Server error.*{expect}"):
                client.request("127.0.0.1", orch.port, command, params, text)
        self.assertEqual(orch.synth.calls, [])

    def test_reply_defaults_and_bad_wav_metadata(self):
        reply = orchestrator_reply({"robot_id": "r", "request_id": "q", "command": ["TTS"], "parameters": {},
                                    "content": "hi"}, FakeSynth())
        self.assertEqual((reply["robot_id"], reply["request_id"], reply["command"]), ("r", "q", []))
        with self.assertRaisesRegex(ValueError, "does not match"):
            client.decode_wav(reply["content"], {**reply["parameters"], "sample_rate": 16000})

    @unittest.skipUnless(LIBRARY_TTS.exists(), "Library-Book-Robot is not next to this repo")
    def test_the_server_teams_own_client_works_against_our_server(self):
        orch = self.serve()
        spec = importlib.util.spec_from_file_location("library_tts", LIBRARY_TTS)
        library_tts = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(library_tts)
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            out = Path(d) / "hello.wav"
            status = library_tts.main(["--host", "127.0.0.1", "--port", str(orch.port), "--text", "Hello there",
                                       "--output", str(out)])
            with wave.open(str(out)) as w:
                self.assertEqual((w.getnchannels(), w.getsampwidth(), w.getframerate()), (1, 2, 22050))
        self.assertEqual(status, 0)


class TestEngine(AIServerCase):
    def test_probe_passes_a_server_that_refuses_an_invalid_emotion_setting(self):
        orch = self.serve()
        eng = AIServerTTSEngine()
        self.assertEqual((eng.emotion_supported, eng.accepts_phoneme_hints, eng.native_sample_rate, eng.emotional),
                         (True, True, 22050, True))
        self.assertEqual(len(orch.synth.calls), 1)                      # the invalid probe was refused before any synthesis
        self.assertEqual(orch.synth.calls[0]["text"], "Hello.")
        self.assertEqual((orch.synth.calls[0]["vector"], orch.synth.calls[0]["alpha"]), NEUTRAL)
        self.assertEqual(eng.address, f"127.0.0.1:{orch.port}")

    def test_a_server_that_ignores_emotion_gets_a_warning_and_no_hints(self):
        for name, echo in ((None, False), ("kokoro", False), ("indextts", True)):   # the last one echoes "indextts" back
            server = other_engine_server(name, echo)
            self.addCleanup(stop, server)
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                eng = AIServerTTSEngine("127.0.0.1", server.server_address[1])
            self.assertFalse(eng.emotion_supported, (name, echo))
            self.assertFalse(eng.accepts_phoneme_hints)
            self.assertIn("accepted an invalid emotion setting", err.getvalue())
            self.assertIn("NO emotion", err.getvalue())

    def test_no_address_and_unreachable(self):
        config.AI_SERVER_HOST = ""
        with self.assertRaisesRegex(RuntimeError, "No AI server address.*AI_HOST"):
            AIServerTTSEngine()
        with self.assertRaisesRegex(RuntimeError, "Could not reach the AI server at 127.0.0.1:9.*--tts indextts"):
            AIServerTTSEngine("127.0.0.1", 9)

    def test_server_error_becomes_a_clear_runtime_error(self):
        self.serve()
        eng = AIServerTTSEngine()
        with self.assertRaisesRegex(RuntimeError, "AI server TTS request failed.*emo_vector"):
            eng.synthesize_emotion("hi", [0.5] * 3, 0.5)
        with self.assertRaisesRegex(ValueError, "1 to 2000"):
            eng.synthesize_emotion("x" * 2001, VECTORS["calm"], 0.5)

    def test_registry_and_default(self):
        self.assertEqual(config.TTS_ENGINE, "aiserver")
        self.serve()
        self.assertIs(engines.engine_for("en-NZ", "aiserver"), engines.engine_for("en-US", "aiserver"))
        with self.assertRaises(NotImplementedError):
            engines.engine_for("zh-CN", "aiserver")


class TestRendering(AIServerCase):
    def test_one_request_per_clause_with_its_own_emotion_and_kupu_hints(self):
        orch = self.serve()
        plan = plan_speech(frame_for("{happy}Kia ora, my whānau are here, {sad}but my nan could not come.",
                                     "person:aroha"), KGContext(demo_graph()))
        with tempfile.TemporaryDirectory() as d:
            res = render_speech(plan, AIServerTTSEngine(), Path(d) / "x.wav")
            self.assertGreater(res.duration_s, 0.5)
        clauses = orch.synth.calls[1:]                                 # the first call is the probe
        self.assertEqual([c["text"] for c in clauses],
                         [f"<Kia ora|{PHONEMES['ck:kia-ora']}>, my <whānau|{PHONEMES['ck:whanau']}> are here,",
                          "but my nan could not come."])
        self.assertEqual([(c["vector"], c["alpha"]) for c in clauses],
                         [(VECTORS["happy"], ALPHAS["happy"]),                       # first clause: full strength
                          (VECTORS["sad"], round(ALPHAS["sad"] * transition_scale(CANONICAL_VA["happy"], CANONICAL_VA["sad"]), 3))])

    def test_no_hints_when_the_server_is_not_indextts(self):
        server = other_engine_server("kokoro")
        self.addCleanup(stop, server)
        with contextlib.redirect_stderr(io.StringIO()):
            eng = AIServerTTSEngine("127.0.0.1", server.server_address[1])
        plan = plan_speech(frame_for("My whānau are here.", "person:aroha"), KGContext(demo_graph()))
        seen = []
        eng.synthesize_emotion = lambda text, vec, alpha, lang="EN": (seen.append(text), (b"\0\0" * 2205, 24000))[1]
        with tempfile.TemporaryDirectory() as d:
            render_speech(plan, eng, Path(d) / "x.wav")
        self.assertEqual(seen, ["My whānau are here."])                 # no <word|phonemes> to be read aloud


class SlowWhenSad(FakeSynth):
    """A synth that really uses the emotion: a sad vector makes the clip 1.5 times longer."""

    def __call__(self, text, vector, alpha, lang="EN"):
        sr, pcm = super().__call__(text, vector, alpha, lang)
        return (sr, pcm + pcm[:len(pcm) // 2 // 2 * 2]) if vector[2] >= 0.5 else (sr, pcm)


class TestConformanceCheck(AIServerCase):
    def run_check(self, host, port):
        lines = []
        results = check.run_checks(host, port, reps=2, say=lines.append)
        return dict(results), "\n".join(lines)

    def test_a_server_that_uses_emotion_passes_every_check(self):
        orch = Orchestrator(SlowWhenSad())
        self.addCleanup(orch.close)
        results, text = self.run_check("127.0.0.1", orch.port)
        self.assertTrue(all(results.values()), text)
        self.assertEqual(len(results), 5)
        self.assertIn("identical audio", text)                  # the fake is deterministic

    def test_a_server_that_validates_but_ignores_emotion_fails_only_the_sound_check(self):
        orch = Orchestrator(FakeSynth())
        self.addCleanup(orch.close)
        results, _ = self.run_check("127.0.0.1", orch.port)
        failed = [name for name, ok in results.items() if not ok]
        self.assertEqual(len(failed), 1)
        self.assertIn("emotion changes the sound", failed[0])

    def test_a_server_that_echoes_the_parameters_fails_the_refusal_and_sound_checks(self):
        server = other_engine_server("indextts", echo=True)
        self.addCleanup(stop, server)
        results, text = self.run_check("127.0.0.1", server.server_address[1])
        self.assertTrue(results["answers a TTS request"])
        self.assertEqual(sum(1 for name, ok in results.items() if not ok and name.startswith("refuses")), 3)
        self.assertFalse(all(results.values()))
        self.assertIn("it is not reading the field", text)

    def test_unreachable_server_stops_after_the_first_check(self):
        results, _ = self.run_check("127.0.0.1", 9)
        self.assertEqual(list(results.values()), [False])

    def test_exit_status(self):
        orch = Orchestrator(SlowWhenSad())
        self.addCleanup(orch.close)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(check.main(["--host", "127.0.0.1", "--port", str(orch.port), "--reps", "1"]), 0)
        self.assertIn("ALL CHECKS PASSED", out.getvalue())
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(check.main(["--host", "127.0.0.1", "--port", "9"]), 1)
        self.assertIn("NOT READY", out.getvalue())


class TestCommandLine(AIServerCase):
    def test_pipeline_speaks_through_the_ai_server(self):
        orch = Orchestrator()
        self.addCleanup(orch.close)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            pipeline_main(["--testset", str(temp_csv(CSV)), "--only", "a2,a6", "--out", tempfile.mkdtemp(),
                           "--ai-host", "127.0.0.1", "--ai-port", str(orch.port)])
        out = buf.getvalue()
        self.assertIn(f"AI server 127.0.0.1:{orch.port} (emotion settings read)", out)
        texts = [c["text"] for c in orch.synth.calls]
        self.assertIn(f"My <whānau|{PHONEMES['ck:whanau']}> is coming to the <hui|{PHONEMES['ck:hui']}> tomorrow.", texts)
        self.assertIn("Hello,", texts)
        self.assertIn("but my nan could not come.", texts)


if __name__ == "__main__":
    unittest.main()
