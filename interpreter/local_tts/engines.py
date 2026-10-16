"""Which engine speaks which language. Engines are loaded on first use and then reused."""
from typing import Optional

from interpreter.local_tts.config import KOKORO_VOICES, MODELS_DIR, TTS_ENGINE, TTS_ENGINES, VOICES

_ENGINES: dict = {}


def engine_for(lang: str, engine: Optional[str] = None):
    """The engine for `lang`: `engine` ("kokoro" or "piper"), else config.TTS_ENGINE."""
    name = engine or TTS_ENGINE
    if name not in TTS_ENGINES:
        raise ValueError(f"engine {name!r} not in {TTS_ENGINES}")
    if lang == "zh-CN":
        raise NotImplementedError("Mandarin voice is not implemented yet")
    voices = KOKORO_VOICES if name == "kokoro" else VOICES
    if lang not in voices:
        raise ValueError(f"no {name} voice configured for {lang!r}; have {sorted(voices)}")
    if (name, lang) not in _ENGINES:
        if name == "kokoro":
            from interpreter.local_tts.kokoro_engine import KokoroEngine
            _ENGINES[name, lang] = KokoroEngine(KOKORO_VOICES[lang])
        else:
            from interpreter.local_tts.piper_engine import PiperEngine
            _ENGINES[name, lang] = PiperEngine(MODELS_DIR / f"{VOICES[lang]}.onnx")
    return _ENGINES[name, lang]
