"""Settings for local speech output."""
from dataclasses import dataclass
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
# Which local engine speaks: "kokoro" (more natural; the default) or "piper" (the original, a deep male voice).
TTS_ENGINE = "kokoro"
TTS_ENGINES = ("kokoro", "piper")
# Kokoro voice per language. The first letter is the accent: b = British, a = American. There is no NZ voice, so en-NZ
# uses the best-graded British voice (bf_emma, B-); af_heart is Kokoro's best overall (A). Swap by ear.
KOKORO_VOICES = {"en-NZ": "bf_emma", "en-US": "af_heart"}
# Piper voice per language; files live in MODELS_DIR. zh-CN is reserved (no voice in either engine).
VOICES = {"en-NZ": "en_GB-alan-medium", "en-US": "en_US-lessac-medium"}
DEFAULT_VOICE = VOICES["en-NZ"]
RESPELL_STYLE = "hyphen"              # "hyphen" -> "fah-no", "joined" -> "fahno"
BASE_PEAK = 0.7                       # whole waveform is normalised to this fraction of full scale


@dataclass(frozen=True)
class RenderConfig:
    respell_style: str = RESPELL_STYLE
    base_peak: float = BASE_PEAK

    def __post_init__(self) -> None:
        if self.respell_style not in ("hyphen", "joined"):
            raise ValueError(f"respell_style={self.respell_style!r} must be 'hyphen' or 'joined'")
