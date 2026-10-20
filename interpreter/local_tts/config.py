"""Settings for local speech output."""
import os
from dataclasses import dataclass
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
# Which engine speaks: "aiserver" (the default: IndexTTS on the AI server, requested over the orchestrator protocol),
# "indextts" (the same IndexTTS as a service on this machine, see below), "kokoro" (natural voice, emotion only through
# our rate/pitch/volume) or "piper" (the original deep voice). IndexTTS is the one with real emotion input: Jay's scoring
# on 2026-10-19 gave it 19/26 sentences and "sounded way better than kokoro".
TTS_ENGINE = "aiserver"
TTS_ENGINES = ("aiserver", "indextts", "kokoro", "piper")
# "aiserver" sends each clause to the AI server's orchestrator (TCP, the protocol of the Library-Book-Robot repo; see
# interpreter/ai_server/), where IndexTTS runs. The address comes from AI_HOST and AI_PORT as in that repo.
AI_SERVER_HOST = os.environ.get("AI_HOST", "")
AI_SERVER_PORT = int(os.environ.get("AI_PORT", "7898"))
# Speech-to-text (the live demo) may be a different orchestrator instance from the text-to-speech one: on 2026-10-20 STT answered
# on 7898 while 8791 served only TTS. Defaults: the same host, port 7898.
AI_STT_HOST = os.environ.get("AI_STT_HOST", "")          # empty = use the AI_HOST address
AI_STT_PORT = int(os.environ.get("AI_STT_PORT", "7898"))
AI_SERVER_TIMEOUT = 300       # seconds; a long clause on a busy server can be slow
AI_SERVER_VOICE = "af_heart"  # for servers whose TTS has voices; IndexTTS clones its reference voice instead
# IndexTTS-2.5 runs in its own venv (it needs its own torch build) as a small HTTP service (indextts_server.py). With the
# default local URL the engine starts it automatically; point INDEXTTS_URL at another machine to use a remote one. The
# venv, weights and git checkout are the ones set up for the bake-off (interpreter/bakeoff/setup.sh).
_BAKEOFF = Path(__file__).resolve().parent.parent / "bakeoff"
# Where the service is. Set the INDEXTTS_URL environment variable (or run_pipeline --indextts-url) to use one that is already
# running, for example on the AI server through an SSH tunnel; the pipeline then never starts a local copy.
# 8790, not 8765: the ChatBox graph viewer uses 8765.
INDEXTTS_URL = os.environ.get("INDEXTTS_URL", "http://127.0.0.1:8790")
INDEXTTS_AUTOSTART = "INDEXTTS_URL" not in os.environ
INDEXTTS_PYTHON = _BAKEOFF / "venvs" / "indextts" / "bin" / "python"
INDEXTTS_REPO = _BAKEOFF / "repos" / "index-tts"
INDEXTTS_MODEL_DIR = _BAKEOFF / "models" / "indextts-2.5"
INDEXTTS_HF_HOME = _BAKEOFF / "models" / "hf"
# Whose voice is cloned: a short neutral clip made with Kokoro (bf_emma, Apache-2.0), kept in the repo so a fresh clone
# (for example on the AI server) has it.
INDEXTTS_REFERENCE = Path(__file__).resolve().parent / "assets" / "neutral_ref.wav"
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
