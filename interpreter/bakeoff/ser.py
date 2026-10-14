"""Speech-emotion recognition over every rendered clip. Run inside venvs/ser (funasr + emotion2vec_plus_large).
A ROUGH PROXY: the model is trained on human and pseudo-labelled speech, not on TTS, and has no 'excited' class.
    venvs/ser/bin/python ser.py [--device cpu|cuda]  ->  out/ser.json  {engine: {file: {label, scores}}}"""
import argparse
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"
MODEL = "emotion2vec/emotion2vec_plus_large"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--engines", help="comma-separated engine dirs under out/ (default: all)")
    args = ap.parse_args()
    from funasr import AutoModel
    model = AutoModel(model=MODEL, hub="hf", device=args.device, disable_update=True)
    only = set(args.engines.split(",")) if args.engines else None
    result = {}
    for engine_dir in sorted(p for p in OUT.iterdir() if p.is_dir() and (only is None or p.name in only)):
        wavs = sorted(engine_dir.glob("*.wav"))
        if not wavs:
            continue
        result[engine_dir.name] = {}
        for wav in wavs:
            r = model.generate(str(wav), granularity="utterance", extract_embedding=False)[0]
            scores = {lab.split("/")[-1]: round(float(s), 4) for lab, s in zip(r["labels"], r["scores"])}
            result[engine_dir.name][wav.name] = {"label": max(scores, key=scores.get), "scores": scores}
            print(engine_dir.name, wav.name, result[engine_dir.name][wav.name]["label"], flush=True)
    (OUT / "ser.json").write_text(json.dumps({"model": MODEL, "results": result}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
