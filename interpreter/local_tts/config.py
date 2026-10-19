"""Settings for local speech output."""
from dataclasses import dataclass
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
# Which engine speaks: "indextts" (real emotion input; the default since Jay's scoring on 2026-10-19: 19/26 sentences
# passed and it "sounded way better than kokoro"; runs as a separate service, see below), "kokoro" (natural voice,
# emotion only through our rate/pitch/volume) or "piper" (the original deep voice).
TTS_ENGINE = "indextts"
TTS_ENGINES = ("kokoro", "indextts", "piper")
# IndexTTS-2.5 runs in its own venv (it needs its own torch build) as a small HTTP service (indextts_server.py). With the
# default local URL the engine starts it automatically; point INDEXTTS_URL at another machine to use a remote one. The
# venv, weights and git checkout are the ones set up for the bake-off (interpreter/bakeoff/setup.sh).
_BAKEOFF = Path(__file__).resolve().parent.parent / "bakeoff"
INDEXTTS_URL = "http://127.0.0.1:8790"   # not 8765: the ChatBox graph viewer uses that port
INDEXTTS_PYTHON = _BAKEOFF / "venvs" / "indextts" / "bin" / "python"
INDEXTTS_REPO = _BAKEOFF / "repos" / "index-tts"
INDEXTTS_MODEL_DIR = _BAKEOFF / "models" / "indextts-2.5"
INDEXTTS_HF_HOME = _BAKEOFF / "models" / "hf"
INDEXTTS_REFERENCE = _BAKEOFF / "reference" / "neutral_ref.wav"   # whose voice is cloned (Kokoro bf_emma, neutral)
# Send phoneme hints for the kupu IndexTTS speaks (indextts_phonemes.py; drafts to be checked by a te reo speaker).
INDEXTTS_KUPU_HINTS = True
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
