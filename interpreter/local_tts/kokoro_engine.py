"""Kokoro-82M engine (Apache-2.0). `kokoro` is imported lazily, like piper. Kokoro has no emotion input of its own;
our planner's rate, pitch and volume are applied on top by render.py, exactly as for Piper."""


class KokoroEngine:
    native_sample_rate = 24000
    base_length_scale = 1.0          # Kokoro takes a speed factor instead; speed = 1 / length_scale

    def __init__(self, voice: str = "bf_emma", device=None) -> None:
        from kokoro import KPipeline   # lazy: optional dependency
        self.voice = voice
        self._pipe = KPipeline(lang_code=voice[0], repo_id="hexgrad/Kokoro-82M", device=device)

    def synthesize(self, text: str, length_scale: float) -> bytes:
        import numpy as np
        chunks = [r.audio.detach().cpu().numpy() for r in self._pipe(text, voice=self.voice, speed=1 / length_scale)
                  if r.audio is not None]
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype="float32")
        return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
