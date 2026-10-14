"""Per-emotion prosody profiles for Task 1. Pure functions, no project imports.

PROFILES are STARTING VALUES TO TUNE BY EAR (see run_local --sweep), not findings: nobody has
checked them against listeners yet.
"""
import math

# emotion -> (rate, pitch, volume) multipliers at full strength
PROFILES = {
    "neutral":    (1.00, 1.00, 1.00),
    "happy":      (1.15, 1.18, 1.10),
    "excited":    (1.28, 1.28, 1.20),
    "calm":       (0.92, 0.95, 0.90),
    "sad":        (0.78, 0.82, 0.80),
    "apologetic": (0.88, 0.90, 0.85),
    "angry":      (1.18, 1.12, 1.30),
    "anxious":    (1.25, 1.15, 0.95),
    "surprised":  (1.10, 1.30, 1.10),
    "confused":   (0.92, 1.08, 0.95),
}

# (lo, hi) per multiplier, applied after scaling
CLAMPS = {"rate": (0.6, 1.6), "pitch": (0.75, 1.35), "volume": (0.5, 1.5)}


def strength(expressiveness: float, valence: float, arousal: float, confidence: float) -> float:
    """How much of the full profile to apply: the dial x how intense the emotion is x how sure we are."""
    intensity = min(1.0, math.hypot(valence, arousal))
    return expressiveness * (0.4 + 0.6 * intensity) * (0.5 + 0.5 * confidence)


def scaled(label: str, s: float) -> tuple:
    """(rate, pitch, volume) for `label` at strength `s`: m -> 1 + (m - 1) * s, then clamped."""
    out = []
    for m, (lo, hi) in zip(PROFILES[label], CLAMPS.values()):
        out.append(max(lo, min(hi, 1 + (m - 1) * s)))
    return tuple(out)
