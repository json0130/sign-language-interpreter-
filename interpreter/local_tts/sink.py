"""A pipeline speech sink that renders each SpeechPlan to a WAV with the local TTS (and optionally plays it)."""
from pathlib import Path

from interpreter.contracts import SpeechPlan
from interpreter.local_tts.playback import play
from interpreter.local_tts.render import RenderResult, render_speech


class LocalSpeechSink:
    def __init__(self, out_dir, engine=None, play_audio: bool = False) -> None:
        """engine=None picks the engine from each plan's voice language (see render_speech)."""
        self.out_dir, self.engine, self.play_audio = Path(out_dir), engine, play_audio
        self.received: list = []
        self.results: list = []

    def speak(self, plan: SpeechPlan) -> RenderResult:
        dest = self.out_dir / plan.voice.language
        dest.mkdir(parents=True, exist_ok=True)
        result = render_speech(plan, self.engine, dest / f"{plan.frame_id}.wav")
        self.received.append(plan)
        self.results.append(result)
        if self.play_audio:
            play(result.path)
        return result
