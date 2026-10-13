"""Listening test for Pepper's voice. Run with --dry-run to rehearse without a robot:
    python3 -m interpreter.robot.listening_test --rater NAME [--host IP | --dry-run] [--seed N] [--no-input] [--out-dir DIR]
"""
import argparse
import csv
import random
import re
import string
from datetime import datetime
from pathlib import Path

from interpreter.robot.items import build_items
from interpreter.robot.pepper import PepperOutput
from interpreter.robot.transport import DryRunTransport, QiTransport

BANNER = """\
This test is for hearing NZ English listeners, to tune Pepper's voice.
The eye-colour cue must be evaluated with Deaf users separately, and only after consultation.
Rate 1-5 (blank = no answer). At each prompt: Enter = continue, r = replay, s = skip. Ctrl-C stops safely.
"""
SCALES = ("naturalness", "emotion_match", "intelligibility", "colour_fit")
HEADER = ("timestamp", "rater", "item_id", "presented_label", "variant_label", *SCALES, "comment")
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "listening_results"


def shuffled(item, rater: str, seed) -> list:
    variants = list(item.variants)
    random.Random(f"{seed}-{rater}-{item.id}").shuffle(variants)
    return variants


def _rating(ask, scale: str) -> str:
    while True:
        r = ask(f"  {scale} (1-5, blank to skip): ").strip()
        if r in ("", "1", "2", "3", "4", "5"):
            return r
        print("  enter 1-5 or leave blank")


def run_session(items, output, rater: str, seed, ask, writer) -> int:
    """Play and rate every variant; returns the number of rows written. Always stops the robot at the end."""
    rows = 0
    try:
        for item in items:
            print(f"\n=== {item.title}\n{item.prompt}")
            for letter, v in zip(string.ascii_uppercase, shuffled(item, rater, seed)):
                if ask(f"\n[{letter}] Enter = play, s = skip: ").strip().lower() == "s":
                    continue
                while True:
                    output.play_raw(v.payload, v.eye)
                    if ask("  Enter = rate, r = replay: ").strip().lower() != "r":
                        break
                scores = [_rating(ask, s) for s in SCALES]
                writer.writerow([datetime.now().isoformat(timespec="seconds"), rater, item.id, letter,
                                 v.label, *scores, ask("  comment: ").strip()])
                rows += 1
    except (KeyboardInterrupt, EOFError):
        print("\nStopped early.")
    finally:
        output.emergency_stop()
    return rows


def smoke(items, output) -> int:
    n = 0
    for item in items:
        for v in item.variants:
            output.play_raw(v.payload, v.eye)
            n += 1
    return n


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rater", required=True, help="a pseudonym or ID, not a real name")
    ap.add_argument("--host")
    ap.add_argument("--port", type=int, default=9559)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-input", action="store_true", help="play every variant without prompts or CSV (smoke check)")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)
    if bool(args.host) == args.dry_run:
        ap.error("give exactly one of --host IP or --dry-run")

    transport = DryRunTransport() if args.dry_run else QiTransport(args.host, args.port)
    output = PepperOutput(transport)
    items = build_items()
    print(BANNER)
    output.start()

    if args.no_input:
        try:
            n = smoke(items, output)
        finally:
            output.emergency_stop()
        calls = len(transport.calls) if args.dry_run else "n/a (real robot)"
        print(f"smoke: {len(items)} items, {n} variants played, {calls} transport calls recorded")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"\W+", "_", args.rater)
    path = out_dir / f"{safe}-{datetime.now():%Y%m%d-%H%M%S}.csv"
    with open(path, "w", newline="", encoding="utf-8", buffering=1) as f:   # line-buffered: rows survive a crash
        writer = csv.writer(f)
        writer.writerow(HEADER)
        rows = run_session(items, output, args.rater, args.seed, input, writer)
    print(f"\n{rows} ratings written to {path}")


if __name__ == "__main__":
    main()
