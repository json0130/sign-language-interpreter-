"""Listening-test pack for hearing NZ English listeners, built from the mock frames and the Task 1 and
NAOqi adapter code (so changes there flow through). Payloads are NAOqi tagged strings."""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from interpreter.baseline import baseline_task1
from interpreter.context import StaticContext
from interpreter.contracts import EyeColour
from interpreter.lexicon import lookup
from interpreter.mocks.mock_recogniser import load_frames
from interpreter.task1 import EYE_PALETTE, plan_speech
from interpreter.tts.naoqi import naoqi_respelling, render

FRAMES = Path(__file__).resolve().parent.parent / "mocks" / "frames"
NEUTRAL_EYE = EyeColour(rgb=EYE_PALETTE["neutral"])


@dataclass(frozen=True)
class Variant:
    label: str
    payload: str
    eye: Optional[EyeColour] = None


@dataclass(frozen=True)
class Item:
    id: str
    title: str
    prompt: str
    variants: List[Variant]


def tags(rate: int = 100, pitch: int = 100, vol: int = 72) -> str:
    """NAOqi opening tags (same layout as the adapter's header)."""
    return f"\\rst\\\\rspd={rate}\\\\vct={pitch}\\\\vol={vol}\\"


def close() -> str:
    return "\\rst\\"


def _raw(text: str, **kw) -> str:
    return f"{tags(**kw)} {text} {close()}"


def _plan_variants(frame) -> List[Variant]:
    out = []
    for label, plan in (("baseline", baseline_task1(frame)), ("rules", plan_speech(frame, StaticContext()))):
        out.append(Variant(label, render(plan).payload, plan.eye))
    return out


def build_items(frames_dir=None) -> List[Item]:
    frames = {f.frame_id: f for f in load_frames(frames_dir or FRAMES, "sign_to_speech")}
    items = [
        Item("emotion-excited", "Excited speaker (f-001)",
             "Which delivery sounds more like an excited person, and is it still natural?",
             _plan_variants(frames["f-001"])),
        Item("emotion-sad", "Sad speaker (f-003)",
             "Which delivery sounds more like a sad person, and is it still natural?",
             _plan_variants(frames["f-003"])),
        Item("emotion-angry", "Angry speaker (f-004)",
             "Which delivery sounds more like an angry person, and is it still natural?",
             _plan_variants(frames["f-004"])),
    ]

    whanau, hui = lookup("whānau"), lookup("hui")
    sentence = "My {} are coming to the {} tomorrow."
    forms = {"respelled": (naoqi_respelling(whanau), naoqi_respelling(hui)),
             "ascii": ("whanau", "hui"),
             "hyphen": (whanau.lower(), hui.lower()),
             "caps": (whanau, hui)}
    items.append(Item("pronounce-whanau", "Pronunciation of whānau and hui",
                      "Which version sounds closest to how these kupu Māori should be said? Is it intelligible?",
                      [Variant(k, _raw(sentence.format(*v)), NEUTRAL_EYE) for k, v in forms.items()]))

    w = naoqi_respelling(whanau)
    around = {"none": ("", ""), "emph1": ("\\emph=1\\ ", " \\emph=0\\"), "emph2": ("\\emph=2\\ ", " \\emph=0\\"),
              "emulated": ("\\rspd=85\\\\vct=115\\ ", " \\rspd=100\\\\vct=100\\")}
    items.append(Item("emphasis-whanau", "Emphasis on whānau",
                      "Which version puts natural, noticeable stress on the word without sounding odd?",
                      [Variant(k, _raw(f"My {a}{w}{b} are coming to the {naoqi_respelling(hui)} tomorrow."), NEUTRAL_EYE)
                       for k, (a, b) in around.items()]))

    text = "We will start the meeting in five minutes."
    ranges = {"neutral": {}, "rate120": {"rate": 120}, "rate80": {"rate": 80},
              "pitch110": {"pitch": 110}, "pitch90": {"pitch": 90}}
    items.append(Item("prosody-range", "How far can rate and pitch move?",
                      "Which versions still sound natural, and where does it start to sound wrong?",
                      [Variant(k, _raw(text, **kw), NEUTRAL_EYE) for k, kw in ranges.items()]))

    clarify = plan_speech(frames["f-005"], StaticContext())
    items.append(Item("clarify-cue", "Clarification request (f-005)",
                      "Does this sound like a polite request to repeat, clearly the robot and not the speaker? "
                      "(The amber pulse is a placeholder; judge colour with Deaf users separately.)",
                      [Variant("amber-pulse", render(clarify).payload, clarify.eye)]))
    return items
