"""Piper (piper-tts) engine. `piper` is imported lazily so the rest of the package needs nothing installed."""
from pathlib import Path


class PiperEngine:
    def __init__(self, model_path):
        from piper import PiperVoice, SynthesisConfig   # lazy: optional dependency
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Piper model not found: {model_path}")
        self._voice = PiperVoice.load(model_path)
        self._cfg = SynthesisConfig
        self.native_sample_rate = self._voice.config.sample_rate
        self.base_length_scale = self._voice.config.length_scale

    def synthesize(self, text: str, length_scale: float) -> bytes:
        chunks = self._voice.synthesize(text, self._cfg(length_scale=length_scale))
        return b"".join(c.audio_int16_bytes for c in chunks)

