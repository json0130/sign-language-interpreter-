"""Speech from the AI server: one TTS request per clause over the orchestrator protocol (client.py).

The request carries the clause's emotion as extra parameters, which an IndexTTS-backed server understands:
    command ["TTS"], content = the text (at most 2000 characters), parameters:
        voice       a voice name for servers whose TTS has voices (config.AI_SERVER_VOICE, default af_heart)
        engine      "indextts"  (asks for IndexTTS)
        emo_vector  8 weights [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
        emo_alpha   0..1, scales the vector
        lang        "EN"
    reply parameters: format "wav", sample_rate, channels, dtype, and "engine" = what actually spoke.
A server that does not know the emotion parameters would speak with its default voice and no emotion, and would read
IndexTTS pronunciation hints out loud. So on connecting, one short test request finds out what the server really is: only
a server that answers engine "indextts" gets emotion hints and kupu phoneme hints; anything else gets a clear warning.
"""
import sys
from typing import List, Optional, Tuple

from interpreter.ai_server import client
from interpreter.local_tts import config

NEUTRAL = ([0, 0, 0, 0, 0, 0, 0, 0.8], 0.5)
MAX_TEXT = 2000


class AIServerTTSEngine:
    emotional = True
    base_length_scale = 1.0

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None, timeout: Optional[float] = None,
                 probe: bool = True) -> None:
        self.host = host or config.AI_SERVER_HOST
        self.port = int(port or config.AI_SERVER_PORT)
        self.timeout = timeout or config.AI_SERVER_TIMEOUT
        if not self.host:
            raise RuntimeError("No AI server address. Set AI_HOST (and AI_PORT, default 7898) or pass --ai-host; or use "
                               "--tts indextts / --tts kokoro for a voice on this machine.")
        self.server_engine: Optional[str] = None       # what the server says spoke, from the probe
        self.native_sample_rate = 22050                # replaced by the probe's answer
        self.accepts_phoneme_hints = False             # true only for a server that reported "indextts"
        if probe:
            self._probe()

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"

    def _probe(self) -> None:
        pcm, rate, params = self._request("Hello.", *NEUTRAL)
        self.native_sample_rate = rate
        self.server_engine = params.get("engine")
        self.accepts_phoneme_hints = self.server_engine == "indextts"
        if not self.accepts_phoneme_hints:
            print(f"warning: the AI server at {self.address} answered with {self.server_engine or 'its default voice'}, "
                  "not IndexTTS. Emotion settings are probably ignored (the speech will have no emotion), and kupu "
                  "pronunciation hints are switched off.", file=sys.stderr, flush=True)

    def _request(self, text: str, vector: List[float], alpha: float, lang: str = "EN") -> Tuple[bytes, int, dict]:
        if not 0 < len(text) <= MAX_TEXT:
            raise ValueError(f"TTS text must be 1 to {MAX_TEXT} characters, got {len(text)}")
        parameters = {"voice": config.AI_SERVER_VOICE, "engine": "indextts", "emo_vector": vector,
                      "emo_alpha": alpha, "lang": lang}
        try:
            response = client.request(self.host, self.port, "TTS", parameters, text, timeout=self.timeout)
        except (OSError, EOFError) as exc:
            raise RuntimeError(f"Could not reach the AI server at {self.address} ({exc}). Check AI_HOST / AI_PORT, that "
                               "the server is running and the port is open, or use --tts indextts / --tts kokoro.") from exc
        except ValueError as exc:
            raise RuntimeError(f"AI server TTS request failed: {exc}") from exc
        pcm, rate = client.decode_wav(response.get("content"), response["parameters"])
        return pcm, rate, response["parameters"]

    def synthesize_emotion(self, text: str, vector: List[float], alpha: float, lang: str = "EN") -> Tuple[bytes, int]:
        """Mono 16-bit PCM and its sample rate for `text` spoken with the given emotion."""
        pcm, rate, _ = self._request(text, vector, alpha, lang)
        return pcm, rate

    def synthesize(self, text: str, length_scale: float) -> bytes:
        """Neutral speech, for callers that only know the plain engine interface (length_scale is ignored)."""
        return self.synthesize_emotion(text, *NEUTRAL)[0]
