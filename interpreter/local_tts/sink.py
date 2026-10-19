"""A pipeline speech sink that renders each SpeechPlan to a WAV with the local TTS (and optionally plays it)."""
from pathlib import Path

from interpreter.contracts import SpeechPlan
from interpreter.local_tts.config import INDEXTTS_REFERENCE, KOKORO_VOICES, TTS_ENGINE, VOICES
from interpreter.local_tts.engines import engine_for
from interpreter.local_tts.playback import play
from interpreter.local_tts.render import RenderResult, render_speech


class LocalSpeechSink:
    def __init__(self, out_dir, engine=None, play_audio: bool = False, tts=None) -> None:
        """engine: a ready engine object, used for every plan. Otherwise `tts` ("kokoro" or "piper", default
        config.TTS_ENGINE) picks the engine per plan language."""
        self.out_dir, self.engine, self.play_audio = Path(out_dir), engine, play_audio
        self.tts = tts or TTS_ENGINE
        self.received: list = []
        self.results: list = []

    def speak(self, plan: SpeechPlan) -> RenderResult:
        dest = self.out_dir / plan.voice.language
        dest.mkdir(parents=True, exist_ok=True)
        engine = self.engine or engine_for(plan.voice.language, self.tts)
        result = render_speech(plan, engine, dest / f"{plan.frame_id}.wav")
        self.received.append(plan)
        self.results.append(result)
        if self.play_audio:
            play(result.path)
        return result

    def voice_name(self, lang: str) -> str:
        """e.g. 'kokoro bf_emma' (for the trace)."""
        if self.engine is not None:
            return f"{type(self.engine).__name__} {getattr(self.engine, 'voice', '')}".strip()
        if self.tts == "indextts":
            return f"indextts (voice cloned from {Path(INDEXTTS_REFERENCE).name})"
        return f"{self.tts} {(KOKORO_VOICES if self.tts == 'kokoro' else VOICES)[lang]}"
