"""Live demo: speak into the microphone and hear the interpretation, spoken with emotion.

    export AI_HOST=172.24.47.255 AI_PORT=8791            # text-to-speech (IndexTTS); speech-to-text is on AI_STT_PORT (7898)
    interpreter/.venv/bin/python -m interpreter.run_live [--listener person:aroha] [--emotion auto|sad|...] [--mic NAME]

    microphone -> speech-to-text (AI server) -> emotion (grammar guard, then text rules) -> KG word choice for the listener
               -> Task 1 plan (one emotion per clause) -> IndexTTS on the AI server -> speaker

Press Enter, speak, and stop talking (it stops by itself after about a second of silence; --push = Enter to stop). Or type a
sentence to skip the microphone. Commands:  /listener ID   /emotion NAME|auto   /again   /who   /help   /quit
The speaker stands in for the signer: with no sign recogniser connected, the emotion comes from the words (text rules) unless
you force one with /emotion. Words that name a kupu Māori ("family", "meeting", "food") are tagged so a listener who knows te reo
hears the kupu (try /listener person:aroha).
"""
import argparse
import re
import time
from pathlib import Path
from typing import List, Optional

from interpreter.ai_server import stt
from interpreter.concepts import CONCEPTS, gloss
from interpreter.contracts import EMOTIONS, LANGS
from interpreter.emotion_flow import SpeakerState
from interpreter.kg.demo import demo_graph
from interpreter.kg.graph import Graph
from interpreter.live import audio
from interpreter.local_tts import config
from interpreter.mocks.stub_sign_generator import StubSignGenerator
from interpreter.run_local import expressiveness_arg
from interpreter.run_pipeline import Tracer, build_pipeline, describe_graph, quiet_library_noise
from interpreter.task1 import Task1Config
from interpreter.testset import make_row

HERE = Path(__file__).resolve().parent
HELP = ("Enter = record (speak, then stop talking) | type a sentence = use it instead of the microphone | /listener ID | "
        "/emotion NAME|auto | /again | /level (test the microphone) | /who | /quit")


def auto_concepts(text: str, lang: str = "en-NZ") -> str:
    """Space-separated ck: slugs for the English words in `text` that have a kupu Māori ('family' -> ck:whanau), so a listener
    who knows te reo hears the kupu. A plain word match: 'work' in 'it does not work' is tagged too."""
    words = set(re.findall(r"[a-z']+", text.lower()))
    return " ".join(slug for slug in CONCEPTS if (gloss(slug, lang) or "").lower() in words)


def make_frame(n: int, text: str, listener: str, signer: str, emotion: Optional[str]):
    """A frame for one utterance. With no forced emotion the recogniser's emotion is 'none' (confidence 0), so the text rules
    decide; a forced emotion is used at confidence 0.8."""
    row = make_row(f"live-{n:03d}", text, emotion or "neutral", 0.8 if emotion else 0.0, listener, auto_concepts(text),
                   signer, input_confidence=0.9)
    return row.frame


def hear(args, say=print) -> Optional[str]:
    """Record from the microphone and return the transcript, or None if nothing usable was said."""
    if args.push:
        pcm = audio.record_push_to_talk(args.mic)
    else:
        say("  listening... speak now, then stop talking")
        pcm = audio.record_utterance(args.mic, threshold=getattr(args, "speech_threshold", args.min_rms))
    return transcribe_pcm(args, pcm, say)


def transcribe_pcm(args, pcm: Optional[bytes], say=print) -> Optional[str]:
    if not pcm:
        say("  (did not hear anything)")
        return None
    level = stt.rms(pcm)
    say(f"  level {level:.0f} (needs {args.min_rms:.0f}), {len(pcm) / 2 / stt.SAMPLE_RATE:.1f} s")
    if level < args.min_rms:
        say("  (too quiet: check the microphone with --mic NAME; see `pactl list short sources`)")
        return None
    t0 = time.time()
    text = stt.transcribe(args.stt_host or args.ai_host or config.AI_STT_HOST or config.AI_SERVER_HOST, args.stt_port,
                          pcm)
    say(f"  speech-to-text took {time.time() - t0:.1f} s")
    return text or None


class Session:
    """The demo's state: who is listening, any forced emotion, and the pipeline (built once)."""

    def __init__(self, args, graph: Graph, sink, play) -> None:
        self.args, self.graph, self.sink, self.play = args, graph, sink, play
        self.listener, self.emotion, self.n, self.last_wav = args.listener, args.emotion, 0, None
        self.trace = Tracer(not args.quiet)
        self.pipe = build_pipeline(graph, sink, StubSignGenerator(), Task1Config(expressiveness=args.expressiveness),
                                   args.target_lang, self.trace, SpeakerState() if args.carry else None)

    def speak(self, text: str, say=print) -> None:
        self.n += 1
        frame = make_frame(self.n, text, self.listener, self.args.signer, self.emotion)
        t0 = time.time()
        plan = self.pipe.process(frame)
        if not getattr(self.sink, "results", None):
            return
        result = self.sink.results[-1]
        self.last_wav = result.path
        say(f"  speech {result.duration_s:.1f} s of audio, made in {time.time() - t0:.1f} s: \"{plan.utterance}\"")
        self.play(result.path)

    def command(self, line: str, say=print) -> bool:
        """Handle a /command. Returns False to quit."""
        word, _, rest = line[1:].partition(" ")
        rest = rest.strip()
        if word in ("quit", "q", "exit"):
            return False
        if word == "listener":
            if rest not in self.graph.nodes:
                say(f"  unknown person {rest!r}; people in the graph: {', '.join(k for k, v in self.graph.nodes.items() if v.kind == 'person')}")
            else:
                self.listener = rest
                say(f"  listener is now {rest}")
        elif word == "emotion":
            if rest in ("", "auto"):
                self.emotion = None
                say("  emotion: automatic (from the words)")
            elif rest in EMOTIONS:
                self.emotion = rest
                say(f"  emotion forced to {rest}")
            else:
                say(f"  unknown emotion; use auto or one of: {', '.join(EMOTIONS)}")
        elif word == "again":
            if self.last_wav:
                self.play(self.last_wav)
            else:
                say("  nothing to repeat yet")
        elif word == "who":
            describe_graph(self.graph)
        elif word == "level":
            say("  speak for 3 seconds...")
            level = audio.measure_level(self.args.mic, 3.0)
            say(f"  level {level:.0f}; speech counts above {getattr(self.args, 'speech_threshold', self.args.min_rms):.0f}")
        else:
            say("  " + HELP)
        return True


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listener", default="person:visitor_01", help="who hears it (a person in the KG; default: knows no te reo)")
    ap.add_argument("--signer", default="person:aroha", help="who is 'signing' (the speaker standing in for the signer)")
    ap.add_argument("--emotion", default=None, choices=EMOTIONS, help="force one emotion (default: from the words)")
    ap.add_argument("--mic", metavar="NAME", help="part of the microphone's PulseAudio name (see `pactl list short sources`)")
    ap.add_argument("--out-device", metavar="NAME", help="part of the speaker's PulseAudio name (see `pactl list short sinks`)")
    ap.add_argument("--push", action="store_true", help="Enter to stop recording, instead of stopping on silence")
    ap.add_argument("--min-rms", type=float, default=stt.MIN_SPEECH_RMS, help="never treat anything quieter than this as speech (default 500)")
    ap.add_argument("--no-calibrate", action="store_true", help="do not measure the room's background noise at start-up (the speech level is then --min-rms)")
    ap.add_argument("--ai-host", metavar="HOST", help="AI server address (default: AI_HOST)")
    ap.add_argument("--ai-port", type=int, metavar="PORT", help="text-to-speech port (default: AI_PORT)")
    ap.add_argument("--stt-host", metavar="HOST", help="speech-to-text address (default: the AI server address)")
    ap.add_argument("--stt-port", type=int, default=config.AI_STT_PORT, metavar="PORT", help="speech-to-text port (default: AI_STT_PORT, else 7898)")
    ap.add_argument("--kg", help="JSON graph saved by Graph.save (default: the demo graph)")
    ap.add_argument("--target-lang", choices=[l for l in LANGS if l != "zh-CN"])
    ap.add_argument("--expressiveness", type=expressiveness_arg, default=Task1Config.expressiveness)
    ap.add_argument("--carry", action="store_true", help="carry the speaker's emotion from one utterance to the next")
    ap.add_argument("--out", default=str(HERE / "audio_out" / "live"))
    ap.add_argument("--wav", type=Path, help="run one turn from a 16 kHz mono WAV instead of the microphone, then exit")
    ap.add_argument("--text", help="run one turn from this text (no microphone, no speech-to-text), then exit")
    ap.add_argument("--no-play", action="store_true", help="make the speech but do not play it")
    ap.add_argument("--quiet", action="store_true", help="skip the stage-by-stage trace")
    args = ap.parse_args(argv)

    quiet_library_noise()
    if args.ai_host or args.ai_port:
        config.AI_SERVER_HOST = args.ai_host or config.AI_SERVER_HOST
        config.AI_SERVER_PORT = args.ai_port or config.AI_SERVER_PORT
    graph = Graph.load(args.kg) if args.kg else demo_graph()
    if args.listener not in graph.nodes:
        raise SystemExit(f"error: unknown listener {args.listener!r}")

    from interpreter.local_tts.engines import engine_for
    from interpreter.local_tts.sink import LocalSpeechSink
    print("connecting to the AI server...", flush=True)
    engine = engine_for(args.target_lang or "en-NZ", "aiserver")        # probes the server: fails early with a clear message
    sink = LocalSpeechSink(args.out, tts="aiserver")
    play = (lambda path: None) if args.no_play else (lambda path: audio.play_wav(path, args.out_device))
    session = Session(args, graph, sink, play)
    print(f"ready: speech-to-text {args.stt_host or config.AI_STT_HOST or config.AI_SERVER_HOST}:{args.stt_port}, "
          f"text-to-speech {engine.address} ({'emotion settings read' if engine.emotion_supported else 'EMOTION IGNORED by the server'}); "
          f"listener {session.listener}; emotion {'automatic' if not session.emotion else session.emotion}")

    if args.text or args.wav:
        text = args.text
        if args.wav:
            import wave
            with wave.open(str(args.wav), "rb") as w:
                if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (1, 2, stt.SAMPLE_RATE):
                    raise SystemExit("error: --wav must be 16 kHz mono 16-bit")
                text = transcribe_pcm(args, w.readframes(w.getnframes()))
        if text:
            print(f"  heard: \"{text}\"")
            session.speak(text)
        return

    args.speech_threshold = args.min_rms
    if not args.no_calibrate and not args.push:
        print("calibrating the microphone: stay quiet for a second...", flush=True)
        ambient = audio.measure_level(args.mic, 1.0)
        args.speech_threshold = audio.speech_threshold(ambient, args.min_rms)
        print(f"  background level {ambient:.0f}, so speech counts above {args.speech_threshold:.0f}")
        if ambient < 5:
            print("  warning: this microphone hears almost nothing (muted, unplugged or very low gain?). Choose another with "
                  "--mic NAME (see `pactl list short sources`) or test with /level.")
    print(HELP)
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if line.startswith("/"):
            if not session.command(line):
                return
            continue
        text = line or hear(args)
        if not text:
            continue
        if not line:
            print(f"  heard: \"{text}\"")
        try:
            session.speak(text)
        except RuntimeError as exc:                      # a server problem should not end the demo
            print(f"  error: {exc}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        import sys
        sys.exit(f"error: {exc}")
