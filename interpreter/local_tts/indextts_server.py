"""IndexTTS-2.5 as a small local HTTP service. It runs inside the IndexTTS venv (IndexTTS needs its own torch build), so
the pipeline (interpreter/.venv) talks to it over HTTP. indextts_engine.py starts it automatically; to run it by hand,
or on another machine (the AI server):

    interpreter/bakeoff/venvs/indextts/bin/python interpreter/local_tts/indextts_server.py \\
        --repo interpreter/bakeoff/repos/index-tts --model-dir interpreter/bakeoff/models/indextts-2.5 \\
        --reference interpreter/bakeoff/reference/neutral_ref.wav [--host 127.0.0.1] [--port 8790]

    GET  /health      -> {"ok": true, "sample_rate": 22050, "reference": "..."}
    POST /synthesize  {"text": str, "emo_vector": [8 floats], "emo_alpha": float, "lang": "EN"} -> audio/wav

It also answers the AI server's orchestrator protocol (the one in the Library-Book-Robot repo: TCP, length header echoed,
JSON body) on --ai-port (default 7899; 0 turns it off), so the pipeline can request speech the same way the robot does:
    {"robot_id", "request_id", "command": ["TTS"], "parameters": {"emo_vector": [8 floats], "emo_alpha": 0..1,
     "lang": "EN"}, "content": text}  ->  {..., "command": [], "parameters": {"format": "wav", "sample_rate", "channels":
     1, "dtype": "int16", "engine": "indextts"}, "content": <base64 WAV>}      (or "command": ["ERR"] and a message)
Both ports share one loaded model; requests are spoken one at a time. The server team's orchestrator can use this as its
TTS backend by forwarding the same request to --ai-port.

The speaker voice is cloned from --reference. No authentication: keep the hosts on 127.0.0.1 unless the network is trusted.
Standalone on purpose (no interpreter imports), so the file can be copied to another machine.
"""
import argparse
import base64
import io
import json
import re
import socketserver
import sys
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer

ORCH_MAX_BYTES = 8 * 1024 * 1024
ORCH_MAX_TEXT = 2000
DEFAULT_VECTOR, DEFAULT_ALPHA = [0, 0, 0, 0, 0, 0, 0, 0.8], 0.5     # IndexTTS's own neutral is "calm"

ANNOTATION = re.compile(r"<([^|>\n]+)\|([^>\n]+)>")     # IndexTTS's own <word|phonemes> format
CHUNK_CHARS = 40                                        # IndexTTS's low-VRAM chunk size


def chunk_text(text: str, max_chars: int = CHUNK_CHARS) -> list:
    """Split at punctuation into pieces of at most `max_chars` spoken characters, never inside a <word|phonemes> hint.

    IndexTTS does this itself on GPUs under 10 GB, but its splitter also cuts at the dots inside a phoneme hint, so the
    hint is no longer recognised and its stress digits are read out ("F AA one"). This replaces it (the server turns
    IndexTTS's own splitting off). A hint counts as the length of its word."""
    saved = []

    def hide(m):
        saved.append(m.group(0))
        return f"\x00{len(saved) - 1}\x00"

    def show(s, words=False):
        return re.sub(r"\x00(\d+)\x00", lambda m: (ANNOTATION.fullmatch(saved[int(m.group(1))]).group(1) if words
                                                     else saved[int(m.group(1))]), s)

    parts = [p for p in re.split(r"(?<=[,.!?;:])", ANNOTATION.sub(hide, text)) if p]
    chunks, current = [], ""
    for part in parts:
        if current and len(show(current + part, words=True)) > max_chars:
            chunks.append(current)
            current = part
        else:
            current += part
    if current:
        chunks.append(current)
    return [show(c).strip() for c in chunks if c.strip()]


def wav_bytes(sample_rate: int, pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


# ---------------------------------------------------------------- the orchestrator protocol
def recv_exact(sock, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(min(size - len(chunks), 65536))
        if not chunk:
            raise EOFError("peer disconnected before the message was complete")
        chunks.extend(chunk)
    return bytes(chunks)


def recv_message(sock) -> dict:
    header = sock.recv(32)
    if not header:
        raise EOFError("peer closed the connection")
    if not header.isdigit() or not 0 < int(header) <= ORCH_MAX_BYTES:
        raise ValueError("invalid message length")
    sock.sendall(header)                                    # echo the length back
    return json.loads(recv_exact(sock, int(header)).decode("utf-8"))


def send_message(sock, message: dict) -> None:
    payload = json.dumps(message, ensure_ascii=False, allow_nan=False).encode("utf-8")
    header = str(len(payload)).encode("ascii")
    sock.sendall(header)
    if recv_exact(sock, len(header)) != header:
        raise ValueError("peer did not echo the message length")
    sock.sendall(payload)


def orchestrator_reply(request: dict, synth) -> dict:
    """The reply to one orchestrator request. `synth(text, vector, alpha, lang) -> (sample_rate, pcm bytes)`."""
    reply = {"robot_id": request.get("robot_id"), "request_id": request.get("request_id")}
    try:
        if request.get("command") != ["TTS"]:
            raise ValueError(f"unsupported command {request.get('command')}; this service only does TTS")
        text, params = request.get("content"), request.get("parameters") or {}
        if not isinstance(text, str) or not 0 < len(text) <= ORCH_MAX_TEXT:
            raise ValueError(f"TTS content must be text of 1 to {ORCH_MAX_TEXT} characters")
        vector = [float(x) for x in params.get("emo_vector", DEFAULT_VECTOR)]
        alpha = float(params.get("emo_alpha", DEFAULT_ALPHA))
        if len(vector) != 8 or not all(0 <= x <= 1 for x in vector) or not 0 <= alpha <= 1:
            raise ValueError("emo_vector needs 8 values and emo_alpha one value, all between 0 and 1")
        sample_rate, pcm = synth(text, vector, alpha, str(params.get("lang", "EN")))
        return {**reply, "command": [], "content": base64.b64encode(wav_bytes(sample_rate, pcm)).decode("ascii"),
                "parameters": {"format": "wav", "sample_rate": sample_rate, "channels": 1, "dtype": "int16",
                               "engine": "indextts", "emo_alpha": alpha}}
    except Exception as e:   # noqa: BLE001  tell the client what went wrong instead of dropping the connection
        return {**reply, "command": ["ERR"], "parameters": {}, "content": f"{type(e).__name__}: {e}"}


def serve_orchestrator(host: str, port: int, synth) -> socketserver.ThreadingTCPServer:
    """Answer orchestrator requests on host:port in a background thread; returns the server (call .shutdown() to stop)."""

    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            try:
                while True:                                     # normally one request per connection
                    send_message(self.request, orchestrator_reply(recv_message(self.request), synth))
            except (EOFError, ConnectionError, ValueError, OSError):
                pass

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    server = Server((host, port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="the index-tts git checkout")
    ap.add_argument("--model-dir", required=True, help="IndexTTS-2.5 weights (with config.yaml)")
    ap.add_argument("--reference", required=True, help="WAV whose voice is cloned")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--ai-host", help="address for the orchestrator-protocol port (default: --host)")
    ap.add_argument("--ai-port", type=int, default=7899, help="orchestrator-protocol port; 0 turns it off")
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    import numpy as np
    import torch
    from indextts.infer_v2_5 import IndexTTS2

    t0 = time.time()
    tts = IndexTTS2(cfg_path=f"{args.model_dir}/config.yaml", model_dir=args.model_dir, use_bf16=True)

    tts.low_vram = False   # chunk_text does its splitting instead (hint-aware, same chunk size)

    def synth(text, vector, alpha, lang="EN"):
        pieces, sr = [], 22050
        for chunk in chunk_text(text):
            torch.manual_seed(0)
            sr, wav = tts.infer(spk_audio_prompt=args.reference, text=chunk, output_path=None, lang=lang,
                                emo_vector=vector, emo_alpha=alpha, use_random=False)
            pieces.append(np.asarray(wav).reshape(-1).astype("<i2"))
        return int(sr), np.concatenate(pieces).tobytes()

    gpu_lock = threading.Lock()                       # the model speaks one request at a time

    def locked_synth(text, vector, alpha, lang="EN"):
        with gpu_lock:
            return synth(text, vector, alpha, lang)

    sample_rate, _ = locked_synth("Hello.", DEFAULT_VECTOR, DEFAULT_ALPHA)   # warm-up, and learn the sample rate
    if args.ai_port:
        serve_orchestrator(args.ai_host or args.host, args.ai_port, locked_synth)
    print(f"[indextts-server] ready in {time.time() - t0:.0f} s: HTTP on {args.host}:{args.port}"
          + (f", orchestrator protocol on {args.ai_host or args.host}:{args.ai_port}" if args.ai_port else "")
          + f", {sample_rate} Hz", flush=True)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                info = {"ok": True, "sample_rate": sample_rate, "reference": args.reference}
                self._send(200, json.dumps(info).encode(), "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/synthesize":
                return self._send(404, b"not found", "text/plain")
            try:
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                vector = [float(x) for x in req["emo_vector"]]
                if len(vector) != 8:
                    raise ValueError("emo_vector needs 8 values")
                sr, pcm = locked_synth(str(req["text"]), vector, float(req["emo_alpha"]), req.get("lang", "EN"))
            except Exception as e:   # noqa: BLE001  report the error to the client instead of dying
                return self._send(400, f"{type(e).__name__}: {e}".encode(), "text/plain")
            self._send(200, wav_bytes(sr, pcm), "audio/wav")

        def log_message(self, fmt, *a):
            print(f"[indextts-server] {fmt % a}", flush=True)

    HTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
