"""Read-only probe of a Pepper: versions, voices, LED groups, Autonomous Life state. Changes nothing.
    python3 -m interpreter.robot.probe --host IP [--port 9559] [--out FILE]"""
import argparse
import json
from pathlib import Path

from interpreter.robot.transport import QiTransport


def probe(t) -> dict:
    tts = lambda m: t.call("ALTextToSpeech", m)
    langs, voices = tts("getAvailableLanguages"), tts("getAvailableVoices")
    groups = t.call("ALLeds", "listGroups")
    nz = [n for n in list(langs) + list(voices) if "nz" in n.lower() or "new zealand" in n.lower()]
    return {
        "systemVersion": t.call("ALSystem", "systemVersion"),
        "robotName": t.call("ALSystem", "robotName"),
        "languages": langs, "voices": voices,
        "language": tts("getLanguage"), "volume": tts("getVolume"),
        "led_groups": groups, "has_FaceLeds": "FaceLeds" in groups,
        "nz_voices": nz,
        "autonomous_life": t.call("ALAutonomousLife", "getState"),
    }


def report(d: dict) -> str:
    lines = [f"System version : {d['systemVersion']}", f"Robot name     : {d['robotName']}",
             f"Languages      : {d['languages']}", f"Voices         : {d['voices']}",
             f"Current lang   : {d['language']}", f"Volume         : {d['volume']}",
             f"LED groups     : {d['led_groups']}",
             f"FaceLeds among LED groups: {'yes' if d['has_FaceLeds'] else 'NO'}",
             "Is any NZ/en-NZ voice available? " + (", ".join(d["nz_voices"]) if d["nz_voices"] else "none found"),
             f"Autonomous Life: {d['autonomous_life']}"]
    if d["autonomous_life"] != "disabled":
        lines.append("WARNING: Autonomous Life is not disabled; it may override eye colours. "
                     "This code never changes it; you decide whether to.")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=9559)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    d = probe(QiTransport(args.host, args.port))
    print(report(d))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(d, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
