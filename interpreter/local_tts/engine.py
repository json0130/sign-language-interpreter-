"""What render.py needs from a speech synthesiser."""
from typing import Protocol


class SynthEngine(Protocol):
    native_sample_rate: int      # Hz of the PCM that synthesize() returns
    base_length_scale: float     # the voice's own neutral length_scale

    def synthesize(self, text: str, length_scale: float) -> bytes:
        """Mono int16 PCM (native byte order) at native_sample_rate. Larger length_scale = slower."""
        ...
