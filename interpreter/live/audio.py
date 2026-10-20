"""Microphone and speaker through PulseAudio (parec / paplay / pactl), as the Library robot does. Standard library only.

The pure part, `utterance_from_chunks`, decides where speech starts and ends in a stream of audio chunks, so it can be tested
without a microphone; the rest wraps the PulseAudio tools.
"""
import shutil
import subprocess
import threading
from typing import Iterable, Iterator, List, Optional

from interpreter.ai_server.stt import MIN_SPEECH_RMS, SAMPLE_RATE, rms

CHUNK_MS = 100


def find_device(kind: str, name: str) -> str:
    """First PulseAudio source or sink ('sources' / 'sinks') whose name contains `name`, ignoring monitors."""
    listing = subprocess.run(["pactl", "list", "short", kind], capture_output=True, text=True, check=True).stdout
    for line in listing.splitlines():
        device = line.split("\t")[1]
        if name.lower() in device.lower() and not device.endswith(".monitor"):
            return device
    raise ValueError(f"No {'microphone' if kind == 'sources' else 'speaker'} matching {name!r}; see `pactl list short {kind}`")


def utterance_from_chunks(chunks: Iterable[bytes], threshold: float = MIN_SPEECH_RMS, rate: int = SAMPLE_RATE,
                          silence_ms: int = 1000, wait_ms: int = 10000, max_ms: int = 30000,
                          preroll_ms: int = 300) -> Optional[bytes]:
    """The speech in a stream of equal-length PCM chunks, or None if nobody spoke.

    Speech starts at the first chunk whose level reaches `threshold` (the `preroll_ms` before it is kept so the first word is
    not clipped) and ends after `silence_ms` of quiet or `max_ms` in total. Gives up after `wait_ms` without speech."""
    preroll: List[bytes] = []
    kept: List[bytes] = []
    started = False
    waited_ms = quiet_ms = total_ms = 0
    for chunk in chunks:
        ms = len(chunk) // 2 * 1000 // rate
        loud = rms(chunk) >= threshold
        if not started:
            if loud:
                started, kept = True, preroll[:] + [chunk]
            else:
                preroll.append(chunk)
                del preroll[: max(0, len(preroll) - max(1, preroll_ms // max(ms, 1)))]
                waited_ms += ms
                if waited_ms >= wait_ms:
                    return None
            continue
        kept.append(chunk)
        total_ms += ms
        quiet_ms = 0 if loud else quiet_ms + ms
        if quiet_ms >= silence_ms or total_ms >= max_ms:
            break
    return b"".join(kept) if started else None


AMBIENT_FACTOR = 2.5      # speech must be this many times louder than the room's background noise


def speech_threshold(ambient: float, minimum: float = MIN_SPEECH_RMS) -> float:
    """The level that counts as speech: AMBIENT_FACTOR times the background noise, but never below `minimum`. A fixed level
    does not work across microphones: the webcam's background reads about 650, a USB microphone's about 1."""
    return max(minimum, AMBIENT_FACTOR * ambient)


def measure_level(mic: Optional[str] = None, seconds: float = 1.0, rate: int = SAMPLE_RATE) -> float:
    """RMS level of what the microphone hears for `seconds`."""
    proc = subprocess.Popen(_record_command(mic, rate), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        return rms(proc.stdout.read(int(rate * 2 * seconds)))
    finally:
        proc.terminate()
        proc.wait()


def _record_command(mic: Optional[str], rate: int) -> List[str]:
    for tool in ("parec", "pactl"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"{tool} not found; install pulseaudio-utils")
    cmd = ["parec", f"--rate={rate}", "--channels=1", "--format=s16le"]
    return cmd + ([f"--device={find_device('sources', mic)}"] if mic else [])


def _stream(proc, rate: int) -> Iterator[bytes]:
    size = rate * 2 * CHUNK_MS // 1000
    while True:
        chunk = proc.stdout.read(size)
        if not chunk:
            return
        yield chunk


def record_utterance(mic: Optional[str] = None, rate: int = SAMPLE_RATE, **kw) -> Optional[bytes]:
    """Listen on the microphone until the speaker finishes (see utterance_from_chunks). Returns PCM or None."""
    proc = subprocess.Popen(_record_command(mic, rate), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        return utterance_from_chunks(_stream(proc, rate), rate=rate, **kw)
    finally:
        proc.terminate()
        proc.wait()


def record_push_to_talk(mic: Optional[str] = None, rate: int = SAMPLE_RATE, prompt: str = "  listening... [Enter] when done ") -> bytes:
    """Record from the microphone until Enter is pressed."""
    proc = subprocess.Popen(_record_command(mic, rate), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    parts: List[bytes] = []
    reader = threading.Thread(target=lambda: parts.extend(_stream(proc, rate)), daemon=True)
    reader.start()
    try:
        input(prompt)
    finally:
        proc.terminate()
        reader.join(timeout=2)
        proc.wait()
    return b"".join(parts)


def play_wav(path, device: Optional[str] = None) -> bool:
    """Play a WAV through PulseAudio (optionally to the sink matching `device`), else the first available player."""
    if device and shutil.which("paplay"):
        subprocess.run(["paplay", f"--device={find_device('sinks', device)}", str(path)], check=False)
        return True
    from interpreter.local_tts.playback import play
    return play(path)
