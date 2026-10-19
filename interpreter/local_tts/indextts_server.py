"""IndexTTS-2.5 as a small local HTTP service. It runs inside the IndexTTS venv (IndexTTS needs its own torch build), so
the pipeline (interpreter/.venv) talks to it over HTTP. indextts_engine.py starts it automatically; to run it by hand,
or on another machine (the AI server):

    interpreter/bakeoff/venvs/indextts/bin/python interpreter/local_tts/indextts_server.py \\
        --repo interpreter/bakeoff/repos/index-tts --model-dir interpreter/bakeoff/models/indextts-2.5 \\
        --reference interpreter/bakeoff/reference/neutral_ref.wav [--host 127.0.0.1] [--port 8765]

    GET  /health      -> {"ok": true, "sample_rate": 22050, "reference": "..."}
    POST /synthesize  {"text": str, "emo_vector": [8 floats], "emo_alpha": float, "lang": "EN"} -> audio/wav

The speaker voice is cloned from --reference. No authentication: keep --host on 127.0.0.1 unless the network is trusted.
Standalone on purpose (no interpreter imports), so the file can be copied to another machine.
"""
import argparse
import io
import json
import sys
import time
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="the index-tts git checkout")
    ap.add_argument("--model-dir", required=True, help="IndexTTS-2.5 weights (with config.yaml)")
    ap.add_argument("--reference", required=True, help="WAV whose voice is cloned")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    import numpy as np
    import torch
    from indextts.infer_v2_5 import IndexTTS2

    t0 = time.time()
    tts = IndexTTS2(cfg_path=f"{args.model_dir}/config.yaml", model_dir=args.model_dir, use_bf16=True)

    def synth(text, vector, alpha, lang="EN"):
        torch.manual_seed(0)
        sr, wav = tts.infer(spk_audio_prompt=args.reference, text=text, output_path=None, lang=lang,
                            emo_vector=vector, emo_alpha=alpha, use_random=False)
        return int(sr), np.asarray(wav).reshape(-1).astype("<i2").tobytes()

    sample_rate, _ = synth("Hello.", [0, 0, 0, 0, 0, 0, 0, 0.8], 0.5)   # warm-up, and learn the sample rate
    print(f"[indextts-server] ready in {time.time() - t0:.0f} s on {args.host}:{args.port}, {sample_rate} Hz", flush=True)

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
                sr, pcm = synth(str(req["text"]), vector, float(req["emo_alpha"]), req.get("lang", "EN"))
            except Exception as e:   # noqa: BLE001  report the error to the client instead of dying
                return self._send(400, f"{type(e).__name__}: {e}".encode(), "text/plain")
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sr)
                w.writeframes(pcm)
            self._send(200, buf.getvalue(), "audio/wav")

        def log_message(self, fmt, *a):
            print(f"[indextts-server] {fmt % a}", flush=True)

    HTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
