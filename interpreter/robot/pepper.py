"""PepperOutput: speak a SpeechPlan and show its eye colour. Speech and LEDs only (see transport.ALLOWED)."""
import itertools
import threading
from dataclasses import replace
from typing import Optional

from interpreter.contracts import EyeColour, SpeechPlan
from interpreter.task1 import EYE_PALETTE
from interpreter.tts.base import Rendered, TTSConfig
from interpreter.tts.naoqi import render

REST = EyeColour(rgb=EYE_PALETTE["neutral"], transition_ms=500)
PULSE_LOW = 0.4   # fraction of brightness at the dim half of a pulse


class PepperOutput:
    def __init__(self, transport, cfg: Optional[TTSConfig] = None):
        self.t = transport
        self.cfg = cfg or TTSConfig()

    def start(self, language: str = "English", volume: float = 0.6) -> None:
        self.t.call("ALTextToSpeech", "setLanguage", language)
        self.t.call("ALTextToSpeech", "setVolume", volume)

    def set_eyes(self, eye: EyeColour) -> None:
        r, g, b = (c / 255 * eye.intensity for c in eye.rgb)
        self.t.call("ALLeds", "fadeRGB", "FaceLeds", r, g, b, eye.transition_ms / 1000)

    def play(self, plan: SpeechPlan) -> Rendered:
        rendered = render(plan, self.cfg)
        self._run(rendered.payload, plan.eye)
        return rendered

    def play_raw(self, payload: str, eye: Optional[EyeColour] = None) -> None:
        self._run(payload, eye)

    def emergency_stop(self) -> None:
        try:
            self.t.call("ALTextToSpeech", "stopAll")
        finally:
            self.set_eyes(REST)

    def _run(self, payload: str, eye: Optional[EyeColour]) -> None:
        stop, pulse = threading.Event(), None
        try:
            if eye is not None:
                self.set_eyes(eye)
                if eye.pattern == "pulse":
                    pulse = threading.Thread(target=self._pulse, args=(eye, stop), daemon=True)
                    pulse.start()
            self.t.call("ALTextToSpeech", "say", payload)
        finally:
            stop.set()
            if pulse is not None:
                pulse.join(timeout=2)
            self.set_eyes(REST)

    def _pulse(self, eye: EyeColour, stop: threading.Event) -> None:
        half = max(0.4, 2 * eye.transition_ms / 1000) / 2
        dim = replace(eye, intensity=eye.intensity * PULSE_LOW)
        for c in itertools.cycle((dim, eye)):
            if stop.wait(half):
                return
            self.set_eyes(c)
