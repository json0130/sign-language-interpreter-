"""Does an AI server's TTS really use the emotion settings? Behaviour checks, not trust in what the reply says:

    python3 -m interpreter.ai_server.check --host HOST --port PORT [--reps 3]

  1. it answers a TTS request with a valid WAV;
  2. it REFUSES invalid emotion settings (an emo_vector of 3 numbers or of text, an emo_alpha of 5). A server that parses the
     fields refuses them; one that only echoes the parameters back (the orchestrator on port 8791 did, 2026-10-20) accepts;
  3. emotion changes the sound: the same sentence spoken as sad takes clearly longer than as neutral, over `reps` repeats
     of each (the server's randomness varies a clip by about 5%; IndexTTS sad is about 1.5 times slower).
Also reported, not scored: whether the same request twice gives identical audio (a fixed seed).
Exit status 0 when every scored check passes. Standard library only.
"""
import argparse
import array
import hashlib
import statistics
import sys
from typing import Callable, List, Tuple

from interpreter.ai_server import client

SENTENCE = "I'm sorry, the meeting has been moved to next week."
NEUTRAL = [0, 0, 0, 0, 0, 0, 0, 0.8]       # [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
SAD = [0, 0, 0.8, 0, 0, 0.2, 0, 0]
MIN_SAD_SLOWDOWN = 1.2                      # sad must average at least this many times neutral's length


def speak(host: str, port: int, vector, alpha, text: str = SENTENCE, timeout: float = 180) -> Tuple[float, str]:
    """(seconds of actual speech, audio fingerprint) for one request. Raises ValueError for a refused request."""
    params = {"voice": "af_heart", "emo_vector": vector, "emo_alpha": alpha, "lang": "EN"}
    response = client.request(host, port, "TTS", params, text, timeout=timeout)
    pcm, rate = client.decode_wav(response["content"], response["parameters"])
    samples = array.array("h")
    samples.frombytes(pcm)
    peak = max((abs(x) for x in samples), default=0)
    loud = [i for i, x in enumerate(samples) if abs(x) > 0.02 * peak]
    return ((loud[-1] - loud[0]) / rate if loud else 0.0), hashlib.sha256(pcm).hexdigest()[:8]


def run_checks(host: str, port: int, reps: int = 3, say: Callable[[str], None] = print) -> List[Tuple[str, bool]]:
    results: List[Tuple[str, bool]] = []

    def record(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, ok))
        say(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))

    try:
        seconds, _ = speak(host, port, NEUTRAL, 0.5)
        record("answers a TTS request", True, f"{seconds:.2f} s of speech")
    except (OSError, EOFError, ValueError) as exc:
        record("answers a TTS request", False, str(exc))
        return results

    for label, vector, alpha in (("an emo_vector of 3 numbers", [0.5, 0.5, 0.5], 0.5),
                                 ("an emo_vector that is text", "banana", 0.5),
                                 ("an emo_alpha of 5", NEUTRAL, 5)):
        try:
            speak(host, port, vector, alpha, "Hello.")
            record(f"refuses {label}", False, "it spoke anyway, so it is not reading the field")
        except ValueError:
            record(f"refuses {label}", True)
        except (OSError, EOFError) as exc:
            record(f"refuses {label}", False, f"connection problem: {exc}")

    neutral, sad = [], []
    for _ in range(reps):
        neutral.append(speak(host, port, NEUTRAL, 0.5)[0])
        sad.append(speak(host, port, SAD, 0.7)[0])
    ratio = statistics.mean(sad) / statistics.mean(neutral) if statistics.mean(neutral) else 0.0
    record(f"emotion changes the sound: sad is at least {MIN_SAD_SLOWDOWN}x slower than neutral", ratio >= MIN_SAD_SLOWDOWN,
           f"sad averaged {statistics.mean(sad):.2f} s, neutral {statistics.mean(neutral):.2f} s over {reps} each = {ratio:.2f}x")

    first, second = speak(host, port, SAD, 0.7)[1], speak(host, port, SAD, 0.7)[1]
    say(f"  INFO  the same request twice gives {'identical audio (a fixed seed)' if first == second else 'different audio (no fixed seed), so scores vary a little between runs'}")
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=client.DEFAULT_PORT)
    ap.add_argument("--reps", type=int, default=3, help="repeats per emotion for the pace check (default 3)")
    args = ap.parse_args(argv)
    print(f"AI server TTS emotion check: {args.host}:{args.port}")
    results = run_checks(args.host, args.port, args.reps)
    failed = [name for name, ok in results if not ok]
    print(f"\n{'ALL CHECKS PASSED: the server uses the emotion settings.' if not failed else 'NOT READY: ' + str(len(failed)) + ' check(s) failed, so speech from this server will have little or no emotion.'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
