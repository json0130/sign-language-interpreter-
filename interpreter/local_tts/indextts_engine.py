"""Client for the IndexTTS-2.5 service (indextts_server.py). Standard library only (urllib).

Unlike Piper and Kokoro, IndexTTS takes the emotion itself (`emotional = True`): render.py hands each clause an emotion
vector and alpha from indextts_emotion.py and does not apply its own rate/pitch trick. If the service is not running and
the URL is local, it is started in the IndexTTS venv and stopped when this process exits.
"""
import atexit
import io
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urlparse

from interpreter.local_tts import config

SERVER = Path(__file__).resolve().parent / "indextts_server.py"
LOG = Path(__file__).resolve().parent.parent / "audio_out" / "indextts_server.log"
NEUTRAL = ([0, 0, 0, 0, 0, 0, 0, 0.8], 0.5)


def gpu_users() -> List[str]:
    """['pid 123 (ollama llama-server): 4666 MiB', ...] for the programs using the GPU; [] if it cannot be read."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    users = []
    for line in out.strip().splitlines():
        pid, _, mib = (x.strip() for x in line.partition(","))
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                argv = fh.read().split(b"\0")
            name = "/".join(argv[0].decode(errors="replace").split("/")[-2:])      # e.g. ollama/llama-server
        except OSError:
            name = "unknown"
        users.append(f"pid {pid} ({name[:60]}): {mib} MiB")
    return users


def start_failure_message(log_text: str, users=None) -> str:
    """Why the service exited while starting, in plain words for the common case (the GPU is full)."""
    tail = log_text.strip().splitlines()[-5:]
    if "OutOfMemoryError" not in log_text:
        return "IndexTTS service exited while starting (or use --tts kokoro):\n  " + "\n  ".join(tail)
    users = gpu_users() if users is None else users
    return ("IndexTTS could not start: the GPU is out of memory (it needs about 5.7 GB). Programs using the GPU now:\n  "
            + ("\n  ".join(users) if users else "(could not read nvidia-smi)")
            + "\nClose the other GPU program and run again (for Ollama: `ollama ps` to see it, `ollama stop <model>` "
              "to unload it; it also unloads itself after about 5 minutes), or run with `--tts kokoro`.")


class IndexTTSEngine:
    emotional = True
    base_length_scale = 1.0

    def __init__(self, url: Optional[str] = None, autostart: bool = True, start_timeout: float = 300) -> None:
        self.url = (url or config.INDEXTTS_URL).rstrip("/")
        self._proc = None
        info = self._health()
        if info is None:
            host = urlparse(self.url).hostname
            if not autostart or host not in ("127.0.0.1", "localhost"):
                raise RuntimeError(f"IndexTTS service not reachable at {self.url}")
            info = self._start(start_timeout)
        self.native_sample_rate = int(info["sample_rate"])
        self.reference = info.get("reference", "")

    # ---------------------------------------------------------------- service
    def _health(self) -> Optional[dict]:
        try:
            with urllib.request.urlopen(f"{self.url}/health", timeout=2) as r:
                return json.load(r)
        except (urllib.error.URLError, OSError, ValueError):
            return None

    def _start(self, timeout: float) -> dict:
        if not config.INDEXTTS_PYTHON.exists():
            raise RuntimeError(f"IndexTTS venv not found at {config.INDEXTTS_PYTHON}; set it up with "
                               "interpreter/bakeoff/setup.sh (the IndexTTS section), or use --tts kokoro")
        port = str(urlparse(self.url).port or 8790)
        cmd = [str(config.INDEXTTS_PYTHON), str(SERVER), "--repo", str(config.INDEXTTS_REPO),
               "--model-dir", str(config.INDEXTTS_MODEL_DIR), "--reference", str(config.INDEXTTS_REFERENCE),
               "--port", port]
        LOG.parent.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "HF_HOME": str(config.INDEXTTS_HF_HOME), "PYTHONPATH": str(config.INDEXTTS_REPO)}
        log = open(LOG, "w")
        self._proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(config.INDEXTTS_REPO), env=env)
        atexit.register(self.close)
        print(f"starting the IndexTTS service (loading takes about a minute; log: {LOG})", end="", flush=True)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._proc.poll() is not None:
                print()
                raise RuntimeError(start_failure_message(LOG.read_text(errors="replace")))
            info = self._health()
            if info is not None:
                print(" ready", flush=True)
                return info
            print(".", end="", flush=True)
            time.sleep(3)
        self.close()
        raise RuntimeError(f"IndexTTS service did not start within {timeout:.0f} s; see {LOG}")

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    # ---------------------------------------------------------------- synthesis
    def synthesize_emotion(self, text: str, vector: List[float], alpha: float, lang: str = "EN") -> Tuple[bytes, int]:
        """Mono 16-bit PCM (native byte order on little-endian machines) and its sample rate."""
        body = json.dumps({"text": text, "emo_vector": vector, "emo_alpha": alpha, "lang": lang}).encode()
        req = urllib.request.Request(f"{self.url}/synthesize", data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                data = r.read()
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"IndexTTS service error: {e.read().decode(errors='replace')}") from e
        with wave.open(io.BytesIO(data), "rb") as w:
            return w.readframes(w.getnframes()), w.getframerate()

    def synthesize(self, text: str, length_scale: float) -> bytes:
        """Neutral speech, for callers that only know the plain engine interface (length_scale is ignored)."""
        return self.synthesize_emotion(text, *NEUTRAL)[0]


def serve_forever() -> None:
    """Start the service (if it is not already running) and keep it up until Ctrl-C, so pipeline runs reuse it
    instead of loading IndexTTS each time:  interpreter/.venv/bin/python -m interpreter.local_tts.indextts_engine"""
    eng = IndexTTSEngine()
    if eng._proc is None:
        print(f"IndexTTS service already running at {eng.url}")
        return
    print(f"IndexTTS service running at {eng.url}; pipeline runs will use it. Ctrl-C to stop.")
    try:
        eng._proc.wait()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        eng.close()


if __name__ == "__main__":
    serve_forever()
