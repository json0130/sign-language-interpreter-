"""Speech-to-text on the AI server, over the orchestrator protocol (client.py). Standard library only.

    command ["STT"], parameters {format: raw_pcm, dtype: int16, sample_rate, channels: 1}, content = base64 PCM
    reply content = the transcript
The server's recogniser invents text for near-silence, so quiet recordings are skipped before sending (the Library robot uses
the same level, 500).
"""
import base64
import math
from array import array
from typing import Optional

from interpreter.ai_server import client

MIN_SPEECH_RMS = 500          # int16 RMS below this counts as silence
SAMPLE_RATE = 16000


def rms(pcm: bytes) -> float:
    """Root-mean-square level of mono int16 PCM (native byte order)."""
    samples = array("h")
    samples.frombytes(pcm[: len(pcm) // 2 * 2])
    return math.sqrt(sum(s * s for s in samples) / len(samples)) if samples else 0.0


def transcribe(host: str, port: int, pcm: bytes, rate: int = SAMPLE_RATE, timeout: float = 120) -> str:
    """The transcript of mono 16-bit PCM. Raises RuntimeError with a message that says what to check."""
    try:
        response = client.request(host, port, "STT", dict(format="raw_pcm", dtype="int16", sample_rate=rate, channels=1),
                                  base64.b64encode(pcm).decode("ascii"), timeout=timeout)
    except (OSError, EOFError) as exc:
        raise RuntimeError(f"Could not reach the AI server for speech-to-text at {host}:{port} ({exc}). Check --stt-host / "
                           "--stt-port (or AI_STT_PORT): the speech-to-text service may be on a different port from the "
                           "text-to-speech one.") from exc
    except ValueError as exc:
        raise RuntimeError(f"AI server speech-to-text failed: {exc}") from exc
    text: Optional[str] = response.get("content")
    if not isinstance(text, str):
        raise RuntimeError("AI server speech-to-text returned no text")
    return text.strip()
