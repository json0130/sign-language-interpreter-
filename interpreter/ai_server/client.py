"""Client for the AI server's orchestrator protocol. Standard library only.

The same wire protocol the Library-Book-Robot repo documents (vendor/ai_server/README.md there): one TCP connection per
request, default port 7898. The sender writes the length of the JSON body as ASCII digits, the receiver echoes them back,
the sender writes the body (UTF-8, at most 8 MiB), and the reply comes back the same way. A request is
    {"robot_id": ..., "request_id": <uuid>, "command": ["TTS"], "parameters": {...}, "content": <text>}
and a successful reply has the same robot_id and request_id, "command": [] and the result in "parameters" / "content"
("command": ["ERR"] and the message in "content" on failure). The address comes from the AI_HOST and AI_PORT
environment variables, as in that repo.
"""
import base64
import binascii
import io
import json
import socket
import uuid
import wave
from typing import Tuple

DEFAULT_PORT = 7898
MAX_MESSAGE_BYTES = 8 * 1024 * 1024


class ProtocolError(ValueError):
    pass


def receive_exact(sock, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(min(size - len(chunks), 65536))
        if not chunk:
            raise EOFError("Peer disconnected before the message was complete")
        chunks.extend(chunk)
    return bytes(chunks)


def send_json(sock, message: dict) -> None:
    payload = json.dumps(message, ensure_ascii=False, allow_nan=False).encode("utf-8")
    if not 0 < len(payload) <= MAX_MESSAGE_BYTES:
        raise ProtocolError("Message exceeds the 8 MiB limit")
    header = str(len(payload)).encode("ascii")
    sock.sendall(header)
    if receive_exact(sock, len(header)) != header:
        raise ProtocolError("Server did not echo the message byte length")
    sock.sendall(payload)


def receive_json(sock) -> dict:
    header = sock.recv(32)
    if not header:
        raise EOFError("Peer closed the connection")
    if not header.isdigit() or not 0 < int(header) <= MAX_MESSAGE_BYTES:
        raise ProtocolError("Invalid message length")
    sock.sendall(header)
    return json.loads(receive_exact(sock, int(header)).decode("utf-8"))


def request(host: str, port: int, command: str, parameters: dict, content: str,
            robot_id: str = socket.gethostname(), timeout: float = 300) -> dict:
    """Send one request and return the full response. Raises OSError/EOFError on connection problems and ValueError on a
    bad response or an ERR response."""
    message = dict(robot_id=robot_id, request_id=str(uuid.uuid4()), command=[command], parameters=parameters,
                   content=content)
    with socket.create_connection((host, port), timeout=timeout) as sock:
        send_json(sock, message)
        response = receive_json(sock)
    if not isinstance(response, dict):
        raise ValueError("Response must be a JSON object")
    for key in ("robot_id", "request_id"):
        if response.get(key) != message[key]:
            raise ValueError(f"Response {key} does not match the request")
    if response.get("command") == ["ERR"]:
        raise ValueError(f"Server error: {response.get('content')}")
    if response.get("command") != [] or not isinstance(response.get("parameters", {}), dict):
        raise ValueError(f"Server returned an incomplete {command} response")
    return response


def decode_wav(encoded, parameters: dict) -> Tuple[bytes, int]:
    """(mono 16-bit PCM bytes, sample rate) from a TTS response's base64 WAV, checked against its metadata."""
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, TypeError, ValueError) as exc:
        raise ValueError("Server returned invalid Base64 audio") from exc
    try:
        with wave.open(io.BytesIO(raw), "rb") as audio:
            if audio.getcomptype() != "NONE":
                raise ValueError("Server WAV is not uncompressed PCM")
            actual = dict(sample_rate=audio.getframerate(), channels=audio.getnchannels(),
                          dtype={1: "uint8", 2: "int16", 4: "int32"}.get(audio.getsampwidth()))
            frames = audio.readframes(audio.getnframes())
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"Server returned an invalid WAV: {exc}") from exc
    for key, value in actual.items():
        if parameters.get(key) != value:
            raise ValueError(f"WAV {key} does not match the response metadata")
    if (actual["channels"], actual["dtype"]) != (1, "int16"):
        raise ValueError(f"expected mono 16-bit audio, got {actual['channels']} channel(s) of {actual['dtype']}")
    return frames, actual["sample_rate"]
