"""Settings for local speech output."""
from dataclasses import dataclass
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
# Piper voice per language; files live in MODELS_DIR. zh-CN is reserved (no voice yet).
# en-NZ uses a British voice as a stand-in until an NZ English voice is chosen.
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
