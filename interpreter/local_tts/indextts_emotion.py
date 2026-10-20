"""Clause emotion -> IndexTTS-2.5 emotion input. Standard library only.

IndexTTS takes an 8-weight emotion vector [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm] and an
emo_alpha (0..1) that scales it. Each of our ten emotions has a vector (mixtures where IndexTTS has no slot) and an alpha.
A clause between two emotions gets a distance-weighted blend, the same way emotion_flow.delivery blends prosody.

Alphas: neutral 0.5, sad 0.7, excited 0.5, angry 0.7 are Jay's picks from the bake-off (1.0 was "a bit too
exaggerated"). Jay found happy at 0.5 sounded neutral and surprised did not sound surprised, so those start higher (0.7
and 0.8). The rest are untuned guesses. Starting values to tune by ear.
"""
import math
from typing import List, Optional, Tuple

from interpreter.emotion import CANONICAL_VA

SLOTS = ("happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm")
VECTORS = {
    "neutral": [0, 0, 0, 0, 0, 0, 0, 0.8],          # IndexTTS's own neutral is "calm"
    "happy": [0.8, 0, 0, 0, 0, 0, 0, 0],
    "excited": [0.7, 0, 0, 0, 0, 0, 0.3, 0],        # no excited slot: happy + surprised
    "calm": [0, 0, 0, 0, 0, 0, 0, 0.9],
    "sad": [0, 0, 0.8, 0, 0, 0.2, 0, 0],
    "apologetic": [0, 0, 0.4, 0, 0, 0.3, 0, 0.4],   # sad + gentle sadness + calm
    "angry": [0, 0.8, 0, 0, 0, 0, 0, 0],
    "anxious": [0, 0, 0, 0.6, 0, 0, 0.2, 0],        # afraid + a little surprised
    "surprised": [0, 0, 0, 0, 0, 0, 0.8, 0],
    "confused": [0, 0, 0, 0.2, 0, 0.1, 0.4, 0],     # no confused slot: approximate
}
# A clause that follows a DIFFERENT emotion is spoken less strongly than the same emotion on its own: "excited, then sad" has
# a gentler sad than a standalone sad (Jay, 2026-10-20: f01 "sudden sad from excited", f05 "sad a bit too much"). The strength
# is multiplied by 1 - TRANSITION_SOFTEN x (distance in valence/arousal from the previous clause), but never below
# TRANSITION_FLOOR. The first clause, and a clause continuing the same emotion, keep their full per-emotion strength.
TRANSITION_SOFTEN = 0.3
TRANSITION_FLOOR = 0.5
# Per-emotion strengths. 2026-10-20 retune after Jay scored the AI server: sad 0.7 -> 0.6 and surprised 0.8 -> 0.65 (a bit
# too much), apologetic 0.5 -> 0.65 and anxious 0.7 -> 0.8 (sounded neutral). One listener, one score each: a first guess.
ALPHAS = {"neutral": 0.5, "sad": 0.6, "excited": 0.5, "angry": 0.7,
          "happy": 0.7, "surprised": 0.65, "calm": 0.5, "apologetic": 0.65, "anxious": 0.8, "confused": 0.5}


def transition_scale(previous: Optional[Tuple[float, float]], current: Tuple[float, float]) -> float:
    """Multiplier for a clause's strength, given the previous clause's (valence, arousal) or None for the first clause."""
    if previous is None:
        return 1.0
    return max(TRANSITION_FLOOR, 1 - TRANSITION_SOFTEN * math.hypot(current[0] - previous[0], current[1] - previous[1]))


def emotion_vector(valence: float, arousal: float) -> Tuple[List[float], float]:
    """(emo_vector, emo_alpha) for a point in valence/arousal. At an emotion's own point it is exactly that emotion's."""
    weights = {l: 1 / ((valence - cv) ** 2 + (arousal - ca) ** 2 + 1e-9) for l, (cv, ca) in CANONICAL_VA.items()}
    total = sum(weights.values())
    vector = [round(sum(w * VECTORS[l][k] for l, w in weights.items()) / total, 4) for k in range(len(SLOTS))]
    alpha = round(sum(w * ALPHAS[l] for l, w in weights.items()) / total, 3)
    return vector, alpha


def describe(vector: List[float]) -> str:
    """'sad 0.8, melancholic 0.2' (the non-zero slots, largest first)."""
    parts = sorted(((v, s) for v, s in zip(vector, SLOTS) if v >= 0.05), reverse=True)
    return ", ".join(f"{s} {v:.2f}" for v, s in parts) or "none"
