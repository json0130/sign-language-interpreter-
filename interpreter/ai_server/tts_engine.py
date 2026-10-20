"""Speech from the AI server: one TTS request per clause over the orchestrator protocol (client.py).

The request carries the clause's emotion as extra parameters, which an IndexTTS-backed server must understand:
    command ["TTS"], content = the text (at most 2000 characters), parameters:
        voice       a voice name for servers whose TTS has voices (config.AI_SERVER_VOICE, default af_heart)
        emo_vector  8 weights [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm], each 0..1
        emo_alpha   0..1, scales the vector
        lang        "EN"
    reply parameters: format "wav", sample_rate, channels, dtype.
A server that does not really read the emotion parameters speaks in its default voice with no emotion, and would read
IndexTTS pronunciation hints out loud. Some servers merely ECHO the request parameters back (one did, on 2026-10-20: its reply
said engine "indextts" because we had asked for it), so the reply cannot be trusted. On connecting, the engine therefore
sends one deliberately invalid emotion setting (an emo_vector of 3 numbers): a server that parses the fields refuses it. A server
that accepts it is not reading them, and gets a clear warning, with kupu phoneme hints switched off.
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
        self.server_engine: Optional[str] = None       # what the server says spoke, if it says (not trusted on its own)
        self.native_sample_rate = 22050                # replaced by the probe's answer
        self.emotion_supported = True                  # false if the server accepts an invalid emo_vector (= ignores it)
        self.accepts_phoneme_hints = False             # true only for a server that passed the check
        if probe:
            self._probe()

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"

    def _probe(self) -> None:
        pcm, rate, params = self._request("Hello.", *NEUTRAL)             # reachability, and the sample rate
        self.native_sample_rate = rate
        self.server_engine = params.get("engine")
        try:
            self._request("Hello.", [0.5, 0.5, 0.5], 0.5)                 # invalid on purpose: 3 numbers, not 8
            self.emotion_supported = False                                # it spoke instead of refusing: fields are ignored
        except RuntimeError:
            self.emotion_supported = True
        self.accepts_phoneme_hints = self.emotion_supported
        if not self.emotion_supported:
            print(f"warning: the AI server at {self.address} accepted an invalid emotion setting (an emo_vector of 3 numbers), "
                  "so it is not reading the emotion fields: the speech will have NO emotion"
                  + (f" (its reply claims engine {self.server_engine!r}, but it only echoes what it was sent)" if self.server_engine else "")
                  + ". Kupu pronunciation hints are switched off. The server's TTS handler must pass emo_vector / emo_alpha "
                  "to IndexTTS; check it with:  python3 -m interpreter.ai_server.check --host HOST --port PORT",
                  file=sys.stderr, flush=True)

    def _request(self, text: str, vector: List[float], alpha: float, lang: str = "EN") -> Tuple[bytes, int, dict]:
        if not 0 < len(text) <= MAX_TEXT:
            raise ValueError(f"TTS text must be 1 to {MAX_TEXT} characters, got {len(text)}")
        parameters = {"voice": config.AI_SERVER_VOICE, "emo_vector": vector, "emo_alpha": alpha, "lang": lang}
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
